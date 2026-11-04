"""
agent.py -- the real agentic tool-calling path.

Earlier versions of this project's docs described the tool-calling
agent as "implemented... but not independently run end-to-end": a
real `langchain_core.tools.Tool` existed (rag/tools.py) and a real
`bind_tools()` call was possible, but nothing in this project had
actually driven a model through the full decide-call-tool-answer
loop. This module closes that gap with a real loop, not a bigger
docstring:

  1. bind the real tool(s) from rag/tools.build_tools() to a real
     ChatAnthropic model via bind_tools()
  2. send the question; if the model's response carries a tool_call,
     that is the model *deciding* to call the tool -- not a keyword
     match, not a hand-rolled router
  3. actually run the real tool (check_live_pattern_state), feed its
     real output back as a ToolMessage, and ask the model for a
     final answer grounded in that real result

What's honestly true about verification: the end-to-end loop against
the REAL Anthropic API has not been run successfully in this sandbox
-- there is no bare ANTHROPIC_API_KEY available here, confirmed
directly (a dummy-key call to the real API returns a clean 401, and
a keyless ChatAnthropic().invoke() returns a connection error -- see
README's "Live-model path" section for how that was checked). What
IS verified, and is a real test of this module's own logic rather
than of Anthropic's API: tests/test_agent.py drives the exact same
run_tool_calling_agent() function against a model double that
returns a real Anthropic-shaped tool_call message, and asserts the
real tool actually executes and its real output is what the second
turn receives. That is the one piece of this loop that is this
project's code to get right or wrong -- whether the live model
*decides* to call the tool is Anthropic's model behavior, not
something this sandbox can fake or needs to.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import Tool

from rag.tools import build_tools

_SYSTEM_PROMPT = (
    "You help a trader understand this project's own real risk-brief "
    "data. You have one tool, check_live_pattern_state, which re-runs "
    "the real pattern detectors and escalation judge live. Call it "
    "ONLY for a question about the CURRENT or LIVE state (e.g. 'is "
    "anything flagged right now', 'should I be notified today'). For "
    "a historical question about a specific past date, answer from "
    "your own knowledge of the conversation instead of calling the "
    "tool -- the tool only ever reports the latest real day's state."
)


@dataclass(frozen=True)
class ToolCallRecord:
    name: str
    args: Dict[str, Any]
    result: str


@dataclass(frozen=True)
class AgentRun:
    final_answer: str
    tool_calls: List[ToolCallRecord] = field(default_factory=list)

    @property
    def model_decided_to_call_tool(self) -> bool:
        return bool(self.tool_calls)


def run_tool_calling_agent(
    llm: Any, question: str, tools: Optional[List[Tool]] = None
) -> AgentRun:
    """Runs the real two-turn tool-calling loop against `llm` (any
    object with the LangChain chat-model `.bind_tools()` / `.invoke()`
    interface -- a real ChatAnthropic in production, a scripted
    double in tests). `llm` is a parameter rather than constructed
    inside this function so a test can pass a model double without
    needing a real API key or monkeypatching an import."""
    tools = tools if tools is not None else build_tools()
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    messages: List[Any] = [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=question)]
    response = llm_with_tools.invoke(messages)

    tool_calls_made: List[ToolCallRecord] = []
    requested_calls = getattr(response, "tool_calls", None) or []
    if not requested_calls:
        return AgentRun(final_answer=str(response.content), tool_calls=[])

    messages.append(response)
    for call in requested_calls:
        tool = tools_by_name[call["name"]]
        result = tool.func(**call.get("args", {}))
        tool_calls_made.append(ToolCallRecord(name=call["name"], args=call.get("args", {}), result=result))
        messages.append(ToolMessage(content=result, tool_call_id=call["id"]))

    final = llm_with_tools.invoke(messages)
    return AgentRun(final_answer=str(final.content), tool_calls=tool_calls_made)
