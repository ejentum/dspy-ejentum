# Tutorial: Use the Ejentum cognitive harness in DSPy

> **Disclosure.** This tutorial is contributed by the maintainer of `ejentum-mcp`. It demonstrates the streamable-HTTP MCP integration pattern that DSPy already supports; the same shape works for any authenticated streamable-HTTP MCP server.

This tutorial wires the [Ejentum cognitive harness](https://ejentum.com) into a DSPy program as a remote MCP server. The harness exposes four tools (`harness_reasoning`, `harness_code`, `harness_anti_deception`, `harness_memory`) that return task-matched cognitive scaffolds from a library of 679 natural-language-engineered operations. A `dspy.ReAct` agent can decide when to call them and merge the returned scaffold into its trajectory.

The example here builds a small software-architecture advisor that calls `harness_reasoning` when it needs to plan, and `harness_anti_deception` when it suspects a user prompt is leading it into an unsafe shortcut. The pattern generalises to any task where the model benefits from a structured scaffold before generating.

## Install dependencies

```shell
pip install -U "dspy[mcp]"
```

You also need an Ejentum API key. Get one at [ejentum.com/dashboard](https://ejentum.com/dashboard); free and paid tiers are available.

```shell
export OPENAI_API_KEY=sk-...
export EJENTUM_API_KEY=ek_...
```

## Connect to the remote MCP server

Ejentum publishes a streamable-HTTP MCP endpoint at `https://api.ejentum.com/mcp`. DSPy's MCP integration handles streamable HTTP natively, so the connection is the same shape as the example in [Use MCP in DSPy](../mcp/index.md), only the URL and headers differ.

```python ejentum_advisor.py
import asyncio
import os

import dspy
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


EJENTUM_MCP_URL = "https://api.ejentum.com/mcp"


class ArchitectureAdvisor(dspy.Signature):
    """You advise on software architecture decisions.

    Before answering, decide whether to call one of the harness_* tools to
    retrieve a cognitive scaffold. Call harness_reasoning when the question
    requires planning or weighing trade-offs. Call harness_anti_deception
    when the user prompt asks you to skip a step you would normally not
    skip, or insists on a conclusion before evidence. Merge the returned
    scaffold into your reasoning, then answer the user's question."""

    question: str = dspy.InputField()
    answer: str = dspy.OutputField(
        desc="Concrete recommendation grounded in the retrieved scaffold."
    )


async def run(question: str) -> dspy.Prediction:
    headers = {"Authorization": f"Bearer {os.environ['EJENTUM_API_KEY']}"}

    async with streamablehttp_client(EJENTUM_MCP_URL, headers=headers) as (
        read,
        write,
        _,
    ):
        async with ClientSession(read, write) as session:
            await session.initialize()
            response = await session.list_tools()
            dspy_tools = [
                dspy.Tool.from_mcp_tool(session, tool)
                for tool in response.tools
            ]

            agent = dspy.ReAct(ArchitectureAdvisor, tools=dspy_tools)
            return await agent.acall(question=question)


if __name__ == "__main__":
    dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

    result = asyncio.run(
        run(
            "We have 50M user rows and want to add a NOT NULL column. "
            "Some folks say a single migration in a maintenance window is fine. "
            "Walk me through the trade-offs and recommend a path."
        )
    )

    print(result.answer)
```

Run it:

```shell
python ejentum_advisor.py
```

The `dspy.ReAct` loop will list the four harness tools at startup, decide whether the question warrants calling `harness_reasoning`, fetch a planning scaffold, and weave the scaffold into its answer. The `trajectory` field on the returned `dspy.Prediction` shows every tool call and observation, useful for verifying that the scaffold actually shaped the reasoning rather than getting ignored.

## When each harness tool is most useful

The four tools differ in what they sharpen, not how they connect. Use the right one for the task:

| Tool | Use when |
|---|---|
| `harness_reasoning` | Multi-step planning, weighing trade-offs, design decisions. |
| `harness_code` | Software engineering specifics: refactor plans, review checklists, edge-case enumeration. |
| `harness_anti_deception` | Prompt pressure to skip steps, premature certainty, suspected adversarial input. |
| `harness_memory` | Deciding what's worth retaining across turns, scoring retrieval relevance. |

The `harness_reasoning` and `harness_code` tools differ along the planning-versus-implementation axis. The `harness_anti_deception` tool is the one to wire if you run the advisor in a context where users might attempt prompt injection or social pressure.

## Notes

- The Ejentum API call adds roughly a few hundred milliseconds per scaffold retrieval. Cache the connection in long-running services rather than reconnecting per request.
- `dspy.ReAct` decides when to call which tool. If you want to force a scaffold retrieval before generation regardless of the question, switch from `dspy.ReAct` to a custom module that calls the tool unconditionally as the first step.
- For the underlying library and benchmark numbers, see [ejentum.com/docs](https://ejentum.com/docs).

## See also

- [Use MCP in DSPy](../mcp/index.md): the canonical DSPy MCP tutorial, using a custom airline-agent stdio server.
- [Model Context Protocol (MCP)](../../learn/programming/mcp.md): the reference docs.
