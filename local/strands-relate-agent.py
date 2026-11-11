# /// script
# requires-python = ">=3.10"
# dependencies = ["strands-agents[openai]==1.63.0", "openai==3.45.0", "relate-db==0.1.2"]
#
# [tool.uv]
# # Relate isn't on PyPI yet: its wheels come from the GitHub release (see local/README.md).
# # The path is relative to where you run uv, so run from the repo root.
# find-links = ["local/wheels"]
# ///
"""Chat with a local copy of the NODES 2026 Aura Agent on Strands Agents, with Relate embedded in-process.

The graph, tools, and command line are in nodes_graph.py. This file adds Strands Agents:
each agent.json tool becomes a Strands tool, and the agent keeps the conversation.

Needs OPENAI_API_KEY.
Usage: uv run local/strands-relate-agent.py [--verbose] ["question"]
  With a question: answer it and exit. Without: chat until Ctrl-C or Ctrl-D.
"""

from openai import OpenAI
from strands import Agent
from strands.models.openai import OpenAIModel
from strands.tools import PythonAgentTool

from nodes_graph import MODEL, VERBOSE, Graph, ToolDef, run_chat, show_step, tool_defs


def strands_tool(tool: ToolDef) -> PythonAgentTool:
    def call(tool_use, **_) -> dict:
        text = tool.run(**tool_use["input"])
        return {"toolUseId": tool_use["toolUseId"], "status": "success", "content": [{"text": text}]}

    schema = {
        "type": "object",
        "properties": {name: {"type": "string", "description": d} for name, d in tool.arguments.items()},
        "required": list(tool.arguments),
    }
    tool_spec = {"name": tool.name, "description": tool.description, "inputSchema": {"json": schema}}
    return PythonAgentTool(tool.name, tool_spec, call)


def make_ask(graph: Graph, spec: dict):
    client = OpenAI()
    tools = tool_defs(
        graph,
        spec,
        embed=lambda model, text: client.embeddings.create(model=model, input=text).data[0].embedding,
        complete=lambda messages: client.chat.completions.create(model=MODEL, messages=messages).choices[0].message.content,
    )
    # callback_handler=None: print only the final answer, not the streamed tokens.
    agent = Agent(
        model=OpenAIModel(model_id=MODEL),
        tools=[strands_tool(t) for t in tools],
        system_prompt=spec["system_prompt"],
        callback_handler=None,
    )

    def ask(question: str) -> str:
        start = len(agent.messages)
        result = agent(question)
        if VERBOSE:
            for m in agent.messages[start:]:
                for block in m["content"]:
                    if "toolUse" in block:
                        show_step(block["toolUse"]["name"], block["toolUse"]["input"])
                    elif "toolResult" in block:
                        show_step(result=" ".join(c.get("text", "") for c in block["toolResult"]["content"]))
        return str(result)

    return ask


if __name__ == "__main__":
    run_chat(make_ask)
