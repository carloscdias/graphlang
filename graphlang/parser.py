import inspect

from langgraph.prebuilt import create_react_agent, ToolNode, tools_condition
from langgraph.prebuilt.chat_agent_executor import AgentState
from langgraph.graph import START, END, StateGraph

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.loading import load_prompt_from_config

from graphlang.graph_parser import Lark_StandAlone, Transformer, v_args

DEFAULT_GRAPH_CONTEXT = {
    'ToolNode':  ToolNode,
    'tools_condition': tools_condition,
}

inline_args = v_args(inline=True)

class Item:
    def __init__(self, **kwargs):
        self.attrs = kwargs.get('attrs', {})

    def apply_attrs(self, attrs: dict):
        self.attrs.update(attrs)

    def __repr__(self):
        return f"Item[{self.attrs}]"

    def execute(self, scope):
        pass


class Attrs(dict):
    def execute(self, scope):
        for k, v in self.items():
            scope.add(k, v)


def new_with_scope(cls, scope):
    opts = {}
    signature = inspect.signature(cls)
    for name, param in signature.parameters.items():
        value = scope.get(name, param.default)
        if value != param.default:
            opts[name] = value
    return cls(**opts)


class Node(Item):
    def __init__(self, name='anon', **kwargs):
        self.name = name
        super().__init__(**kwargs)

    def __repr__(self):
        return f"{self.name}({self.attrs})"

    def execute(self, scope):
        inner_scope = Scope(scope=self.attrs, parent=scope, workflow=scope.workflow)
        class_type = inner_scope.get('node_class', create_react_agent)
        node = new_with_scope(class_type, inner_scope)
        scope.add(self.name, node)
        scope.workflow.add_node(self.name, node)


class NodeGroup(Node):
    def __init__(self, nodes=[], **kwargs):
        self.nodes = nodes
        super().__init__(**kwargs)

    def apply_attrs(self, attrs):
        if isinstance(attrs, list):
            self._apply_multiple(attrs)
        else:
            self._apply_single(attrs)

    def _apply_single(self, attrs):
        for n in self.nodes:
            n.apply_attrs(attrs)

    def _apply_multiple(self, attrs):
        n_nodes = len(self.nodes)
        n_attrs = len(attrs)

        if n_attrs != n_nodes:
            raise Exception(f"Number of nodes ({n_nodes}) and attributes ({n_attrs}) are not the same")

        for i in range(n_nodes):
            self.nodes[i].apply_attrs(attrs[i])

    def __repr__(self):
        return f"Group({self.nodes})[{self.attrs}]"

    def execute(self, scope):
        for n in self.nodes:
            n.execute(scope)


class Model(Item):
    def __init__(self, name='anon', **kwargs):
        self.name = name
        super().__init__(**kwargs)

    def __repr__(self):
        return f"Model{self.attrs}"

    def execute(self, scope):
        inner_scope = Scope(parent=scope, scope=self.attrs, workflow=scope.workflow)
        model_class = inner_scope.get('model_class')
        model = new_with_scope(model_class, inner_scope)
        scope.add(self.name, model)


class Prompt(Item):
    DEFAULT = 'chatmessages'

    def __init__(self, name='anon', **kwargs):
        self.name = name
        super().__init__(**kwargs)

    def __repr__(self):
        return f"Prompt{self.attrs}"

    def execute(self, scope):
        inner_scope = Scope(parent=scope, scope=self.attrs, workflow=scope.workflow)
        cls = None
        if self.attrs.get('_type', Prompt.DEFAULT) == Prompt.DEFAULT:
            cls = ChatPromptTemplate
        else:
            cls = load_prompt_from_config
        prompt = new_with_scope(cls, inner_scope)
        scope.add(self.name, prompt)


class StartNode(Node):
    def __init__(self, node=None, **kwargs):
        self.node = node
        super().__init__(**kwargs)

    def __repr__(self):
        return f"*{self.node.name}"

    def execute(self, scope):
        scope.workflow.add_edge(START, self.node.name)


class Edge(Item):
    def __init__(self, right=None, symbol='?', **kwargs):
        self.left = None
        self.right = right
        self.symbol = symbol
        super().__init__(**kwargs)

    def __repr__(self):
        return f"{self.left} {self.symbol} {self.right}[{self.attrs}]"

    def set_left(self, node):
        self.left = node

    @classmethod
    def add_workflow_edge_many(cls, left, right, workflow, opts={}):
        for l in left:
            for r in right:
                if "condition" in opts:
                    workflow.add_edge(l, r, opts[condition])
                else:
                    workflow.add_edge(l, r)

    @classmethod
    def add_workflow_edge(cls, left, right, workflow, opts={}):
        if isinstance(left, NodeGroup):
            left_names = [l.name for l in left.nodes]
        else:
            left_names = [left.name]
        if isinstance(right, NodeGroup):
            right_names = [r.name for r in right.nodes]
        else:
            right_names = [right.name]
        Edge.add_workflow_edge_many(left_names, right_names, workflow, opts)


class DEdge(Edge):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, symbol='->')

    def execute(self, scope):
        inner_scope = Scope(parent=scope, scope=self.attrs, workflow=scope.workflow)
        condition = inner_scope.get('condition', None)
        condition = inner_scope.get('right', condition)
        opts = {"condition": condition} if condition else {}
        Edge.add_workflow_edge(self.left, self.right, scope.workflow, opts)


class UEdge(Edge):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, symbol='--')

    def execute(self, scope):
        inner_scope = Scope(parent=scope, scope=self.attrs, workflow=scope.workflow)
        condition = inner_scope.get('condition', None)
        condition = inner_scope.get('right', condition)
        opts = {"condition": condition} if condition else {}
        Edge.add_workflow_edge(self.left, self.right, scope.workflow, opts)
        condition = inner_scope.get('left', condition)
        opts = {"condition": condition} if condition else {}
        Edge.add_workflow_edge(self.right, self.left, scope.workflow, opts)


class EdgeGroup(Edge):
    def __init__(self, edges=[], **kwargs):
        self.edges = edges
        super().__init__(**kwargs)

    def apply_attrs(self, attrs):
        if isinstance(attrs, list):
            self._apply_multiple(attrs)
        else:
            self._apply_single(attrs)

    def _apply_single(self, attrs):
        for edge in self.edges:
            edge.apply_attrs(attrs)

    def _apply_multiple(self, attrs):
        n_edges = len(self.edges)
        n_attrs = len(attrs)

        if n_attrs != n_edges:
            raise Exception(f"Number of edges ({n_edges}) and attributes ({n_attrs}) are not the same")

        for i in range(n_edges):
            self.edges[i].apply_attrs(attrs[i])

    def __repr__(self):
        return f"{self.edges}"

    def execute(self, scope):
        for e in self.edges:
            e.execute(scope)


class Graph(Node):
    def __init__(self, stmt_list=[], scope={}, **kwargs):
        self.stmt_list = stmt_list
        self.scope = scope
        super().__init__(**kwargs)

    def __repr__(self):
        return f"{self.name}{self.stmt_list}"

    def execute(self, scope):
        state = scope.get('state_class', AgentState)
        workflow = StateGraph(state)
        inner_scope = Scope(parent=scope, scope=self.attrs, workflow=workflow)
        for stmt in self.stmt_list:
            stmt.execute(inner_scope)
        compiled_graph = new_with_scope(workflow.compile, inner_scope)
        scope.add(self.name, compiled_graph)


class Scope(Item):
    def __init__(self, scope={}, parent=None, workflow=None, **kwargs):
        self.parent = parent
        self.scope = scope
        self.workflow = workflow
        super().__init__(**kwargs)

    def add(self, name, value):
        self.scope[name] = value

    def get(self, key, default = None):
        value = self.scope.get(key, self.parent.get(key, default) if self.parent else default)
        if isinstance(value, Reference):
            value = value(self)
        return value

    def __repr__(self):
        return f"{self.scope}"


class Reference(Item):
    def __init__(self, key=None, **kwargs):
        self.key = key
        super().__init__(**kwargs)

    def __call__(self, scope):
        return scope.get(self.key)

    def __repr__(self):
        return f"Reference({self.key})"


class TreeToGraph(Transformer):

    def __init__(self):
        self.defs = {}
        super().__init__()

    def __default__(self, rule, children, *args):
        return children

    @inline_args
    def start(self, stmt_list):
        main_graph = Graph(name='main', stmt_list=stmt_list)
        self.defs['main'] = main_graph
        return main_graph

    @inline_args
    def start_stmt(self, node):
        return StartNode(node=node)

    @inline_args
    def stmt_list(self, stmt=None, next_stmt=None):
        if not stmt:
            return
        if not next_stmt:
            return [stmt]
        return [stmt, *next_stmt]

    @inline_args
    def only_attr_stmt(self, stmt_type, name, attr_stmt_line={}):
        return stmt_type(name=name, attrs=attr_stmt_line)

    def model(self, children):
        return Model

    def prompt(self, children):
        return Prompt

    @inline_args
    def attr_stmt_line(self, attr={}, next_attr=None):
        if next_attr:
            attr.update(next_attr)
        return attr

    def multiple_attr(self, children):
        return children

    @inline_args
    def graph_stmt(self, name, stmt_list):
        self.defs[name] = Graph(name=name, stmt_list=stmt_list)
        return self.defs[name]

    @inline_args
    def with_attr_stmt(self, node, attrs=None):
        if attrs:
            node.apply_attrs(attrs)
        return node

    @inline_args
    def attr_stmt(self, attr, children = None):
        if children:
            attr.update(children)
        return attr

    @inline_args
    def node_id(self, value):
        if value not in self.defs:
            self.defs[value] = Node(name=value)
        return self.defs[value]

    @inline_args
    def node_id_list(self, node, next_node=None):
        if not next_node:
            return node
        return NodeGroup(nodes=[node, next_node])

    @inline_args
    def edge_stmt(self, node_group, next_edges):
        next_edges[0].set_left(node_group)
        return EdgeGroup(edges=next_edges)

    @inline_args
    def edge_rhs(self, edge_type, node_group, next_edges=None):
        new_node = edge_type(right=node_group)
        if next_edges and len(next_edges) > 0:
            next_edges[0].set_left(node_group)
            return [new_node, *next_edges]
        return [new_node]

    @inline_args
    def attr(self, name, value):
        return Attrs([(name, value)])

    @inline_args
    def reference(self, value):
        return Reference(key=value)

    def ID(self, token):
        return token.value

    def NUMBER(self, token):
        return float(token.value)

    @inline_args
    def string(self, s):
        return s[1:-1].replace('\\"', '"')

    def ESCAPED_STRING(self, s):
        return s.value

    list  = list
    tuple = tuple
    dict  = dict
    key_value = tuple
    uedge = lambda self, _: UEdge
    dedge = lambda self, _: DEdge


def parse_graph(graph_file, context={}):
    parser = Lark_StandAlone(debug=True)
    global_context = DEFAULT_GRAPH_CONTEXT.copy()
    global_context.update(context)
    with open(graph_file) as f:
        tree = parser.parse(f.read())
        transformer = TreeToGraph()
        graph = transformer.transform(tree)
        scope = Scope(scope=global_context)
        graph.execute(scope)
        return scope.get('main')

