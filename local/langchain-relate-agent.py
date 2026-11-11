# /// script
# requires-python = ">=3.10"
# dependencies = ["langchain==1.5.0", "langchain-openai==1.7.4", "relate-db==0.1.2"]
#
# [tool.uv]
# # Relate isn't on PyPI yet: its wheels come from the GitHub release (see local/README.md).
# # The path is relative to where you run uv, so run from the repo root.
# find-links = ["local/wheels"]
# ///
"""Chat with a local copy of the NODES 2026 Aura Agent, built with LangChain on a Relate graph.

The agent's system prompt and tools come from agent/agent.json. Each tool is a LangChain @tool,
named and described as in agent.json, that runs its Cypher on the graph (see nodes_graph.py):
- cypherTemplate: runs the template with the model's parameters.
- similaritySearch: embeds the query with the configured embedding model, then searches the
  vector index.
- text2cypher: the chat model writes the Cypher; if the query fails, it gets the error and one
  more try.
create_agent runs the agent, a checkpointer keeps the conversation, and --verbose streams each
step as it happens.

Needs OPENAI_API_KEY.
Usage: uv run local/langchain-relate-agent.py [--verbose] ["question"]
  With a question: answer it and exit. Without: chat until Ctrl-C or Ctrl-D.
"""

import json
import os
import sys

from langchain.agents import create_agent
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain.embeddings import init_embeddings
from langchain.messages import AIMessage, ToolMessage
from langchain.tools import BaseTool, tool
from langgraph.checkpoint.memory import InMemorySaver

from nodes_graph import (
    MODEL,
    SIMILARITY_SCHEMA,
    TEXT2CYPHER_SCHEMA,
    VERBOSE,
    Graph,
    chat_loop,
    cypher_from_reply,
    load_agent_spec,
    template_schema,
    text2cypher_instructions,
    to_json,
    tools_of_type,
)


def cypher_template_tool(graph: Graph, spec: dict) -> BaseTool:
    @tool(spec["name"], description=spec["description"], args_schema=template_schema(spec, ["name"]))
    def cypher_template(name: str) -> str:
        return graph.run_template(spec, name=name)

    return cypher_template


def similarity_search_tool(graph: Graph, spec: dict) -> BaseTool:
    embeddings = init_embeddings(f"openai:{spec['config']['model']}")

    @tool(spec["name"], description=spec["description"], args_schema=SIMILARITY_SCHEMA)
    def similarity_search(query: str) -> str:
        return graph.vector_search(spec, embeddings.embed_query(query))

    return similarity_search


def text2cypher_tool(graph: Graph, spec: dict, model: BaseChatModel) -> BaseTool:
    @tool(spec["name"], description=spec["description"], args_schema=TEXT2CYPHER_SCHEMA)
    def text2cypher(question: str) -> str:
        messages = [{"role": "system", "content": text2cypher_instructions(spec)}, {"role": "user", "content": question}]
        for _ in range(2):
            cypher = cypher_from_reply(model.invoke(messages).text)
            try:
                return graph.run_generated(cypher)
            except Exception as err:  # Retry once with the error, the way a person would.
                error = str(err)
                messages += [
                    {"role": "assistant", "content": cypher},
                    {"role": "user", "content": f"That failed: {error}. Fix the query."},
                ]
        return to_json({"cypher": cypher, "error": error})

    return text2cypher


def build_agent(graph: Graph, spec: dict, model: BaseChatModel):
    tools = [cypher_template_tool(graph, t) for t in tools_of_type(spec, "cypherTemplate")]
    tools += [similarity_search_tool(graph, t) for t in tools_of_type(spec, "similaritySearch")]
    tools += [text2cypher_tool(graph, t, model) for t in tools_of_type(spec, "text2cypher")]
    return create_agent(model, tools=tools, system_prompt=spec["system_prompt"], checkpointer=InMemorySaver())


def show_step(message) -> None:
    """With --verbose, print a tool call or the start of its result."""
    if isinstance(message, AIMessage):
        for call in message.tool_calls:
            print(f"  → {call['name']}({json.dumps(call['args'], ensure_ascii=False)})")
    elif isinstance(message, ToolMessage):
        text = message.text
        print(f"  ← {text[:300]}{'…' if len(text) > 300 else ''}")


def make_ask(agent):
    """ask(question) for chat_loop: one conversation thread, kept by the checkpointer."""
    config = {"configurable": {"thread_id": "nodes-chat"}}

    def ask(question: str) -> str:
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": question}]}, config, stream_mode="updates", version="v2"
        ):
            if VERBOSE:
                for update in chunk["data"].values():
                    for message in (update or {}).get("messages", []):
                        show_step(message)
        return agent.get_state(config).values["messages"][-1].text

    return ask


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first.")
    spec = load_agent_spec()
    graph = Graph()
    try:
        agent = build_agent(graph, spec, init_chat_model(f"openai:{MODEL}"))
        chat_loop(spec, make_ask(agent))
    finally:
        graph.close()


if __name__ == "__main__":
    main()
