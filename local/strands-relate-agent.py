# /// script
# requires-python = ">=3.10"
# dependencies = ["strands-agents[openai]==1.64.0", "openai==3.49.0", "relate-db==0.1.2"]
#
# [tool.uv]
# # Relate isn't on PyPI yet: its wheels come from the GitHub release (see local/README.md).
# # The path is relative to where you run uv, so run from the repo root.
# find-links = ["local/wheels"]
# ///
"""Chat with a local copy of the NODES 2026 Aura Agent, built with Strands Agents on a Relate graph.

The agent's system prompt and tools come from agent/agent.json. Each tool is a Strands @tool,
named and described as in agent.json, that runs its Cypher on the graph (see nodes_graph.py):
- cypherTemplate: runs the template with the model's parameters.
- similaritySearch: embeds the query with OpenAI (Strands has no embeddings API, so this calls
  the OpenAI client directly), then searches the vector index.
- text2cypher: an agent as a tool. A specialist Strands agent writes the Cypher; if the query
  fails, it gets the error and one more try.
The Strands agent keeps the conversation, and --verbose adds a plugin that prints each tool call.

Needs OPENAI_API_KEY.
Usage: uv run local/strands-relate-agent.py [--verbose] ["question"]
  With a question: answer it and exit. Without: chat until Ctrl-C or Ctrl-D.
"""

import json
import os
import sys

from openai import OpenAI
from strands import Agent, tool
from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent
from strands.models import Model
from strands.models.openai import OpenAIModel
from strands.plugins import Plugin, hook

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


def cypher_template_tool(graph: Graph, spec: dict):
    @tool(name=spec["name"], description=spec["description"], inputSchema=template_schema(spec, ["name"]))
    def cypher_template(name: str) -> str:
        return graph.run_template(spec, name=name)

    return cypher_template


def similarity_search_tool(graph: Graph, spec: dict, openai: OpenAI):
    @tool(name=spec["name"], description=spec["description"], inputSchema=SIMILARITY_SCHEMA)
    def similarity_search(query: str) -> str:
        embedding = openai.embeddings.create(model=spec["config"]["model"], input=query).data[0].embedding
        return graph.vector_search(spec, embedding)

    return similarity_search


def text2cypher_tool(graph: Graph, spec: dict, model: Model):
    @tool(name=spec["name"], description=spec["description"], inputSchema=TEXT2CYPHER_SCHEMA)
    def text2cypher(question: str) -> str:
        # A new agent for each question: tools can run concurrently, and each question gets its
        # own conversation, so the retry sees just this query and its error.
        writer = Agent(model=model, system_prompt=text2cypher_instructions(spec), callback_handler=None)
        prompt = question
        for _ in range(2):
            cypher = cypher_from_reply(str(writer(prompt)))
            try:
                return graph.run_generated(cypher)
            except Exception as err:  # Retry once with the error, the way a person would.
                error = str(err)
                prompt = f"That failed: {error}. Fix the query."
        return to_json({"cypher": cypher, "error": error})

    return text2cypher


class ShowToolCalls(Plugin):
    """With --verbose, print each tool call and the start of its result."""

    name = "show-tool-calls"

    @hook
    def before_tool(self, event: BeforeToolCallEvent) -> None:
        print(f"  → {event.tool_use['name']}({json.dumps(event.tool_use['input'], ensure_ascii=False)})")

    @hook
    def after_tool(self, event: AfterToolCallEvent) -> None:
        text = " ".join(c.get("text", "") for c in event.result["content"])
        print(f"  ← {text[:300]}{'…' if len(text) > 300 else ''}")


def build_agent(graph: Graph, spec: dict, model: Model, openai: OpenAI) -> Agent:
    tools = [cypher_template_tool(graph, t) for t in tools_of_type(spec, "cypherTemplate")]
    tools += [similarity_search_tool(graph, t, openai) for t in tools_of_type(spec, "similaritySearch")]
    tools += [text2cypher_tool(graph, t, model) for t in tools_of_type(spec, "text2cypher")]
    return Agent(
        model=model,
        tools=tools,
        system_prompt=spec["system_prompt"],
        plugins=[ShowToolCalls()] if VERBOSE else [],
        callback_handler=None,  # Print only the final answer, not the streamed tokens.
    )


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first.")
    spec = load_agent_spec()
    graph = Graph()
    try:
        agent = build_agent(graph, spec, OpenAIModel(model_id=MODEL), OpenAI())
        chat_loop(spec, lambda question: str(agent(question)))
    finally:
        graph.close()


if __name__ == "__main__":
    main()
