# GraphLang

**GraphLang** provides a domain-specific language (DSL) to define large language model (LLM) agent graphs. The DSL allows you to structure and manage complex interactions between LLMs and their attributes. This library helps you create agent graphs with flexible syntax and is designed for advanced AI workflows.

## Features

- Define LLM agent graphs with nodes, edges, and attributes
- Use flexible syntax to add models and prompts
- Supports complex attribute assignment via lists, tuples, and dictionaries

## Installation

Install the library using `pip`:

```bash
pip install graphlang
```
## Example

To compile a `.graph` file into a `langgraph` structure, use:

```python
from graphlang.parser import parse_graph

graph = parse_graph(graph_file, {
    'model_class': ChatOllama
})
```

- The second argument is a **context**, which allows you to expose external variables or objects to be used in the `.graph` file, like `"model_class"` in this case, which defines the ChatModel used in all `model` references, or `"state_class"` for States.

In case of a quick react interaction you may also use:

```bash
python -m graphlang <your_graph_file>
```

## Language Syntax

GraphLang is a domain-specific language inspired by the [DOT language](https://graphviz.org/doc/info/lang.html), designed to define LLM agent graphs. Below is a "Hello World" example demonstrating the basic structure:

### Example:

```dot
prompt HelloWorldPrompt {
    messages=[("system", "Hello, World!")];
}

model simple_model {
    model="llama2";
}

node Greeter[state_modifier=HelloWorldPrompt, model=simple_model];

start Greeter;
```

## Quick Syntax Overview

1. **`prompt` Block**: Defines a prompt with messages and input variables for the agent.
   ```dot
   prompt HelloWorldPrompt {
       messages=[("system", "Hello, World!")];
   }
   ```

2. **`model` Block**: Defines the model used by the agent.
   ```dot
   model simple_model {
       model="llama3.1";
   }
   ```

3. **`node` Statement**: Defines an agent (node) with a prompt and a model.
   ```dot
   node Greeter[state_modifier=HelloWorldPrompt, model=simple_model];
   ```

4. **`start` Statement**: Specifies the entry point of the graph.
   ```dot
   start Greeter;
   ```

### Reserved Words

- **`start`**: Specifies the initial node to execute.
- **`node`**: Defines an agent in the graph.
- **`prompt`**: Defines a prompt template for messages and input variables.
- **`model`**: Specifies the model used by the agent.

This structure, based on the DOT language, simplifies the process of defining and executing agent graphs for LLMs.

## Development

For development, install the necessary dependencies:

```bash
pip install graphlang[dev]
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

