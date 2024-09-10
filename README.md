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

## Example Usage

### Example DSL Graph

Here is an example graph defined using the DSL:

```plaintext
start node1;

node node1, node2;
node1 -> node2;

graph my_graph {
    model my_model {
        attribute = 1;
    }
    prompt my_prompt {
        text = "Enter your message.";
    }
}
```

## DSL Syntax

The DSL follows these key rules:

## Development

For development, install the necessary dependencies:

```bash
pip install graphlang[dev]
```

Run tests with:

```bash
pytest
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

