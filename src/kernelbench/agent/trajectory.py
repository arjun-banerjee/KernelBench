"""
Trajectory dataclasses for multi-turn kernel agent runs.

A trajectory captures the full history of one agent run on one problem:
  - Every LLM turn: the input items sent, the structured output items
    received, and every tool call executed within the turn.
  - The final KernelExecResult (or None if the agent never submitted).
  - Run-level metadata (model, backend, timestamps, caps).

Serialization is JSON-only. Non-serializable values (exceptions, torch.dtype,
etc.) are coerced to strings.

Responses-API note
------------------
`KernelTurn.messages_in` and `KernelTurn.response` are both lists of plain
dicts matching OpenAI Responses-API item shapes: role/content messages,
`function_call` items, `function_call_output` items, and `reasoning` items.
The agent stores them via `.model_dump()` so they round-trip cleanly through
JSON without depending on the OpenAI SDK at deserialization time.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from kernelbench.eval import KernelExecResult


# ---------------------------------------------------------------------------
# ToolCall — one invocation of a tool within a turn
# ---------------------------------------------------------------------------


@dataclass
class ToolCall:
    """Record of a single tool invocation."""

    tool_name: str
    args: dict[
        str, Any
    ]  # arguments passed (kernel_code, etc.; large strings are truncated)
    result_text: str  # human/LLM-readable output
    success: bool  # did the tool succeed without error?
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# KernelTurn — one complete LLM response + all tool calls within it
# ---------------------------------------------------------------------------


@dataclass
class KernelTurn:
    """One turn = one LLM response + all tool calls made in that response."""

    turn_id: int
    # Full `input` array sent to the model for this turn (list of Responses-API items).
    messages_in: list[dict[str, Any]]
    # The model's output items for this turn (reasoning, function_call, message, etc.).
    response: list[dict[str, Any]]
    # Tool calls parsed from this response and executed.
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Free-form feedback string (used only on LLM-call failure turns; the
    # regular tool-results flow is captured in `tool_calls` instead).
    feedback_to_model: str = ""
    # Wall-clock time for the LLM call (seconds).
    llm_latency_s: float = 0.0
    # Whether this turn contained a final submission (submit_kernel that
    # produced a real KernelExecResult).
    is_final: bool = False
    # The submitted kernel source at submission time, if any.
    submitted_kernel: str | None = None


# ---------------------------------------------------------------------------
# KernelTrajectory — full run on one problem
# ---------------------------------------------------------------------------


@dataclass
class KernelTrajectory:
    """Full trajectory for one agent run on one problem."""

    # Problem identity
    problem_id: int
    level: int
    problem_name: str

    # Run config
    run_name: str
    model_name: str
    backend: str
    precision: str
    max_turns: int
    max_tool_calls: int
    tools_enabled: list[str]

    # Turns (appended as the agent runs)
    turns: list[KernelTurn] = field(default_factory=list)

    # Final result (set after submit_kernel or when turns exhausted)
    final_result: KernelExecResult | None = None

    # Timestamps (ISO-8601 strings)
    started_at: str = field(
        default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    )
    finished_at: str | None = None

    # Summary stats (filled in by finish())
    total_turns: int = 0
    total_tool_calls: int = 0
    outcome: str = (
        "unknown"  # "correct" | "incorrect" | "compile_fail" | "timeout" | "error"
    )

    # ---------------------------------------------------------------------------

    def add_turn(self, turn: KernelTurn) -> None:
        self.turns.append(turn)
        self.total_turns = len(self.turns)
        self.total_tool_calls = sum(len(t.tool_calls) for t in self.turns)

    def finish(self, result: KernelExecResult | None) -> None:
        """Call when the agent run is complete."""
        self.final_result = result
        self.finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.total_turns = len(self.turns)
        self.total_tool_calls = sum(len(t.tool_calls) for t in self.turns)

        if result is None:
            self.outcome = "error"
        elif not result.compiled:
            self.outcome = "compile_fail"
        elif not result.correctness:
            self.outcome = "incorrect"
        else:
            self.outcome = "correct"

    # ---------------------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""

        def _coerce(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _coerce(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_coerce(v) for v in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                return str(obj)

        return {
            "problem_id": self.problem_id,
            "level": self.level,
            "problem_name": self.problem_name,
            "run_name": self.run_name,
            "model_name": self.model_name,
            "backend": self.backend,
            "precision": self.precision,
            "max_turns": self.max_turns,
            "max_tool_calls": self.max_tool_calls,
            "tools_enabled": self.tools_enabled,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_turns": self.total_turns,
            "total_tool_calls": self.total_tool_calls,
            "outcome": self.outcome,
            "final_result": _coerce(
                self.final_result.model_dump() if self.final_result else None
            ),
            "turns": [
                {
                    "turn_id": t.turn_id,
                    "messages_in": _coerce(t.messages_in),
                    "response": _coerce(t.response),
                    "feedback_to_model": t.feedback_to_model,
                    "llm_latency_s": t.llm_latency_s,
                    "is_final": t.is_final,
                    "submitted_kernel": t.submitted_kernel,
                    "tool_calls": [
                        {
                            "tool_name": tc.tool_name,
                            "args": _coerce(tc.args),
                            "result_text": tc.result_text,
                            "success": tc.success,
                            "metadata": _coerce(tc.metadata),
                        }
                        for tc in t.tool_calls
                    ],
                }
                for t in self.turns
            ],
        }

    def save(self, path: str) -> None:
        """Save trajectory as JSON to the given path."""
        import os

        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> dict[str, Any]:
        """Load a previously saved trajectory as a raw dict for inspection."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
