import sys
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from graphlang.parser import parse_graph

import inspect

def main():
    checkpointer = MemorySaver()
    graph_file = sys.argv[1]
    graph = parse_graph(graph_file, {
        'memory_checkpointer': checkpointer,
        'model_class': ChatOllama,
    })

    config = {
        "configurable": {
            "thread_id": "some_id",
        },
    }

    while True:
        query = input('-> ')
        if query in ['/exit', '/bye']:
            break
        for s in graph.stream({"messages": [HumanMessage(content=query)]}, config, stream_mode="values"):
            message = s["messages"][-1]
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

if __name__ == '__main__':
    main()

