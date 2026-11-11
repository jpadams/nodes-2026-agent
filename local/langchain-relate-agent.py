# /// script
# requires-python = ">=3.10"
# dependencies = ["langchain==1.4.5", "langchain-openai==1.7.1", "relate-db==0.1.2"]
#
# [tool.uv]
# # Relate isn't on PyPI yet: its wheels come from the GitHub release (see local/README.md).
# # The path is relative to where you run uv, so run from the repo root.
# find-links = ["local/wheels"]
# ///
"""Chat with a local copy of the NODES 2026 Aura Agent on LangChain, with Relate embedded in-process.

The graph, tools, and command line are in nodes_graph.py. This file adds LangChain:
each agent.json tool becomes a StructuredTool, and ask() keeps the conversation.

Needs OPENAI_API_KEY.
Usage: uv run local/langchain-relate-agent.py [--verbose] ["question"]
  With a question: answer it and exit. Without: chat until Ctrl-C or Ctrl-D.
"""

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import Field, create_model

from nodes_graph import MODEL, VERBOSE, Graph, ToolDef, run_chat, show_step, tool_defs


def langchain_tool(tool: ToolDef) -> StructuredTool:
    fields = {name: (str, Field(description=d)) for name, d in tool.arguments.items()}
    return StructuredTool.from_function(
        func=tool.run,
        name=tool.name,
        description=tool.description,
        args_schema=create_model(f"{tool.name}_args", **fields),
    )


def make_ask(graph: Graph, spec: dict):
    llm = ChatOpenAI(model=MODEL)
    tools = tool_defs(
        graph,
        spec,
        embed=lambda model, text: OpenAIEmbeddings(model=model).embed_query(text),
        complete=lambda messages: llm.invoke(messages).content,
    )
    agent = create_agent(model=llm, tools=[langchain_tool(t) for t in tools], system_prompt=spec["system_prompt"])
    history = []

    def ask(question: str) -> str:
        nonlocal history
        start = len(history) + 1
        history = agent.invoke({"messages": history + [HumanMessage(question)]})["messages"]
        if VERBOSE:
            for m in history[start:]:
                if isinstance(m, AIMessage):
                    for call in m.tool_calls:
                        show_step(call["name"], call["args"])
                elif isinstance(m, ToolMessage):
                    show_step(result=str(m.content))
        return history[-1].content

    return ask


if __name__ == "__main__":
    run_chat(make_ask)
