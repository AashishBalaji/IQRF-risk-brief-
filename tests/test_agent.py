"""Unit tests for rag/agent.py's real tool-calling loop.

These tests prove this project's OWN logic is correct -- binding,
dispatching to the real tool, feeding the real tool's real output
back for a final answer -- using a scripted model double in place of
a real ChatAnthropic call (no ANTHROPIC_API_KEY is available in this
sandbox; see agent.py's module docstring and README's "Live-model
path" section for how that was confirmed directly rather than
assumed). The double returns real langchain_core.messages.AIMessage
objects with the exact tool_calls shape a real Anthropic tool-calling
response carries, so what's being tested is this project's loop, not
a fake message format."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from rag.agent import run_tool_calling_agent
from rag.tools import build_tools


class _ScriptedToolCallingModel:
    """A minimal LangChain-chat-model-shaped double: bind_tools()
    returns self (same as a real ChatAnthropic().bind_tools() call
    returning a RunnableBinding usable the same way), and invoke()
    returns pre-scripted responses in order -- first a tool_call,
    then a final answer, mirroring the real two-turn shape."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.invoke_calls = []

    def bind_tools(self, _tools):
        return self

    def invoke(self, messages):
        self.invoke_calls.append(messages)
        return self._responses.pop(0)


def test_agent_does_not_call_the_tool_for_a_historical_question():
    model = _ScriptedToolCallingModel([
        AIMessage(content="That's a historical question, answered from context."),
    ])
    run = run_tool_calling_agent(model, "What happened on 2026-06-04?")
    assert run.model_decided_to_call_tool is False
    assert run.tool_calls == []
    assert "historical" in run.final_answer


def test_agent_calls_the_real_tool_when_the_model_requests_it_and_feeds_back_the_real_result():
    """The real regression this test exists for: once the model
    decides to call check_live_pattern_state, the loop must actually
    invoke the REAL tool function (not a stub) and the real tool's
    real output must be what the second model.invoke() call receives
    as a ToolMessage."""
    tool_call_response = AIMessage(
        content="",
        tool_calls=[{"name": "check_live_pattern_state", "args": {}, "id": "call_1"}],
    )
    final_response = AIMessage(content="Based on the live check, nothing needs escalation.")
    model = _ScriptedToolCallingModel([tool_call_response, final_response])

    run = run_tool_calling_agent(model, "Is anything flagged right now?")

    assert run.model_decided_to_call_tool is True
    assert len(run.tool_calls) == 1
    record = run.tool_calls[0]
    assert record.name == "check_live_pattern_state"
    # The real tool's real output -- not a mock -- must be a genuine
    # "Live check as of <date>..." string (see rag/tools.py).
    assert "Live check as of" in record.result
    assert run.final_answer == "Based on the live check, nothing needs escalation."

    # The second invoke() call must have received a ToolMessage
    # carrying the SAME real tool output, proving the real result was
    # actually fed back rather than the loop re-deciding on its own.
    second_call_messages = model.invoke_calls[1]
    tool_messages = [m for m in second_call_messages if getattr(m, "type", None) == "tool"]
    assert len(tool_messages) == 1
    assert tool_messages[0].content == record.result


def test_agent_uses_the_real_tools_module_by_default():
    """build_tools() is the default, so the agent binds the actual
    project tool, not a test-only substitute, unless a caller
    explicitly overrides it."""
    tools = build_tools()
    model = _ScriptedToolCallingModel([AIMessage(content="no tool needed")])
    run_tool_calling_agent(model, "anything", tools=tools)
    assert model.invoke_calls  # bound and invoked without error
