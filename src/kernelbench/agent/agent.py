"""
KernelAgent — multi-turn, tool-using agent for KernelBench, built on the
OpenAI Responses API.

Architecture
------------
- One agent instance handles one (problem, run) pair.
- Each call to run() returns a KernelTrajectory (full history + final result).
- Conversation state is stateless: the full `input` array is resent every turn.
  Reasoning items returned by the model flow back into `input` unchanged so
  the model sees its own chain-of-thought across turns. (The Responses API
  errors if reasoning items are dropped between turns with reasoning models.)
- Tools are declared natively via JSON schema. No XML parsing; the API
  returns structured `function_call` items and we reply with
  `function_call_output` items matched by `call_id`.
- The system prompt is passed as the top-level `instructions` parameter,
  separate from `input`.

Loop, in words:
    instructions := build_system_prompt(...)
    input := [problem_message]
    for turn in range(max_turns):
        response = client.responses.create(instructions, input, tools, ...)
        input += response.output            (preserves reasoning + tool calls)
        for fc in function_calls(response):
            result = execute_tool(fc)
            input.append(function_call_output(fc.call_id, result.output))
            if fc.name == 'submit_kernel' and result is a real KernelExecResult:
                return trajectory
        if no function_calls: break
    return trajectory
"""

from __future__ import annotations

import json
import time
import traceback
from typing import Any

import torch
from openai import OpenAI

from kernelbench.eval import KernelExecResult
from kernelbench.agent.tools import Tool, ToolContext, ToolResult, get_tools
from kernelbench.agent.trajectory import KernelTurn, KernelTrajectory, ToolCall
from kernelbench.agent.prompt_templates import (
    build_system_prompt,
    build_problem_message,
    build_turn_warning_message,
)


class KernelAgent:
    """
    Multi-turn, tool-using agent for a single KernelBench problem.

    The caller is responsible for constructing the OpenAI client. This keeps
    the agent agnostic to Azure/OpenAI-direct/compatible-gateway wiring — the
    caller passes in a configured `OpenAI(...)` instance with whatever
    `api_key` and `base_url` they need.

    Args:
        problem_id:         Integer problem ID.
        level:              Dataset level (1, 2, or 3).
        problem_name:       Human-readable problem name.
        ref_arch_src:       Reference PyTorch model source code.
        client:             An openai.OpenAI instance, pre-configured with
                            api_key and (optionally) base_url.
        model:              Model name to pass to client.responses.create.
        run_name:           Name for this run (used in trajectory metadata).
        tool_names:         Names of tools to enable. None = default set
                            (all except profile_kernel).
        max_turns:          Max number of LLM turns.
        max_tool_calls:     Max total tool calls across all turns.
        backend:            Kernel backend. Default "cuda".
        precision:          Computation precision. Default "fp32".
        device:             torch.device for evaluation.
        build_dir:          CUDA compile cache directory.
        num_correct_trials: Correctness trials for run_correctness / submit.
        num_perf_trials:    Timing trials for submit_kernel.
        timing_method:      Timing method for submit_kernel.
        reasoning_effort:   "minimal" | "low" | "medium" | "high" | None.
                            If set, passed to the API as reasoning.effort.
        warn_turns_remaining: Inject warning when this many turns remain.
        turn_delay_s:       Sleep between turns (for rate-limited APIs).
        verbose:            Verbose logging.
    """

    def __init__(
        self,
        *,
        problem_id: int,
        level: int,
        problem_name: str,
        ref_arch_src: str,
        client: OpenAI,
        model: str,
        run_name: str = "default",
        tool_names: list[str] | None = None,
        max_turns: int = 10,
        max_tool_calls: int = 30,
        backend: str = "cuda",
        precision: str = "fp32",
        device: torch.device | None = None,
        build_dir: str | None = None,
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        timing_method: str = "cuda_event",
        reasoning_effort: str | None = None,
        warn_turns_remaining: int = 2,
        turn_delay_s: float = 0.0,
        verbose: bool = False,
    ) -> None:
        self.problem_id = problem_id
        self.level = level
        self.problem_name = problem_name
        self.ref_arch_src = ref_arch_src
        self.client = client
        self.model = model
        self.run_name = run_name
        self.max_turns = max_turns
        self.max_tool_calls = max_tool_calls
        self.backend = backend
        self.precision = precision
        self.reasoning_effort = reasoning_effort
        self.warn_turns_remaining = warn_turns_remaining
        self.turn_delay_s = turn_delay_s
        self.verbose = verbose

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Resolve tools: `tool_names` filters the registry; get_tools() ensures
        # submit_kernel is always included.
        self.tools: list[Tool] = get_tools(tool_names)
        self.tool_names_enabled: list[str] = [t.name for t in self.tools]
        self.tool_map: dict[str, Tool] = {t.name: t for t in self.tools}

        # ToolContext is constructed once and reused for every tool call this run.
        self.ctx = ToolContext(
            ref_arch_src=ref_arch_src,
            backend=backend,
            precision=precision,
            device=device,
            build_dir=build_dir,
            num_correct_trials=num_correct_trials,
            num_perf_trials=num_perf_trials,
            timing_method=timing_method,
            verbose=verbose,
        )

        # Per-run mutable state.
        self._total_tool_calls: int = 0
        self._final_result: KernelExecResult | None = None

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self) -> KernelTrajectory:
        """Execute the agent loop and return a completed KernelTrajectory."""
        trajectory = KernelTrajectory(
            problem_id=self.problem_id,
            level=self.level,
            problem_name=self.problem_name,
            run_name=self.run_name,
            model_name=self.model,
            backend=self.backend,
            precision=self.precision,
            max_turns=self.max_turns,
            max_tool_calls=self.max_tool_calls,
            tools_enabled=self.tool_names_enabled,
        )

        instructions = build_system_prompt(
            max_turns=self.max_turns,
            max_tool_calls=self.max_tool_calls,
            backend=self.backend,
        )
        problem_msg = build_problem_message(
            ref_arch_src=self.ref_arch_src,
            backend=self.backend,
            precision=self.precision,
        )

        # The input array we resend every turn. It grows monotonically as:
        #   - the model emits reasoning / function_call items (we append all
        #     of response.output)
        #   - we append function_call_output items after each tool executes
        #   - we append our own role=user messages (problem, warnings)
        input_items: list[dict[str, Any]] = [
            {"role": "user", "content": problem_msg},
        ]

        tool_schemas = [t.to_responses_schema() for t in self.tools]

        for turn_idx in range(self.max_turns):
            # Pacing for rate-limited APIs.
            if turn_idx > 0 and self.turn_delay_s > 0:
                time.sleep(self.turn_delay_s)

            turns_remaining = self.max_turns - turn_idx
            tool_calls_remaining = self.max_tool_calls - self._total_tool_calls

            if turns_remaining <= self.warn_turns_remaining and turn_idx > 0:
                input_items.append(
                    {
                        "role": "user",
                        "content": build_turn_warning_message(
                            turns_remaining,
                            tool_calls_remaining,
                        ),
                    }
                )

            # Snapshot the input we're about to send so the trajectory can
            # replay this exact turn. Deep-copy via json round-trip so later
            # mutations of input_items don't leak in.
            messages_in_snapshot = json.loads(json.dumps(input_items))

            # --- LLM call ---
            t0 = time.time()
            try:
                create_kwargs: dict[str, Any] = {
                    "model": self.model,
                    "instructions": instructions,
                    "input": input_items,
                    "tools": tool_schemas,
                }
                if self.reasoning_effort is not None:
                    create_kwargs["reasoning"] = {"effort": self.reasoning_effort}

                response = self.client.responses.create(**create_kwargs)
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                tb = traceback.format_exc()
                print(
                    f"[Agent] LLM call FAILED on turn {turn_idx} "
                    f"(problem {self.problem_id}, level {self.level}):\n"
                    f"{err_msg}\n{tb}"
                )
                failed_turn = KernelTurn(
                    turn_id=turn_idx,
                    messages_in=messages_in_snapshot,
                    response=[],
                    feedback_to_model=f"LLM call failed: {err_msg}",
                    llm_latency_s=time.time() - t0,
                    is_final=False,
                )
                trajectory.add_turn(failed_turn)
                trajectory.finish(self._final_result)
                return trajectory
            llm_latency = time.time() - t0

            # Serialize the model's output items so we can both (a) resend
            # them to the API next turn and (b) store them in the trajectory.
            # model_dump() gives us plain dicts, which is what both consumers
            # want.
            response_items: list[dict[str, Any]] = [
                item.model_dump() for item in response.output
            ]
            input_items.extend(response_items)

            if self.verbose:
                n_fc = sum(
                    1 for it in response_items if it.get("type") == "function_call"
                )

                print(
                    f"\n[Agent] Turn {turn_idx} "
                    f"({llm_latency:.1f}s, {len(response_items)} output items, "
                    f"{n_fc} function calls)"
                )

                for it in response_items:
                    if it.get("type") == "function_call":
                        fc = it.get("function_call", {})
                        print(f"[fn] {fc.get('name')}({fc.get('arguments')})")

            # --- Execute tool calls ---
            function_calls = [
                it for it in response_items if it.get("type") == "function_call"
            ]

            executed_tool_calls: list[ToolCall] = []
            is_final = False
            submitted_kernel: str | None = None

            if not function_calls:
                # Model produced no tool calls. It either finished (unusual
                # without submit_kernel) or is stalling. Record the turn and
                # end the loop — there's nothing to act on.
                turn = KernelTurn(
                    turn_id=turn_idx,
                    messages_in=messages_in_snapshot,
                    response=response_items,
                    tool_calls=[],
                    feedback_to_model="",
                    llm_latency_s=llm_latency,
                    is_final=False,
                )
                trajectory.add_turn(turn)
                if self.verbose:
                    print("[Agent] No tool calls in response — ending loop.")
                break

            for fc in function_calls:
                # Enforce the total tool-call budget. We still need to reply
                # with a function_call_output for every function_call so the
                # API doesn't error on the next turn — so synthesize a
                # budget-exceeded output for any call beyond the cap.
                if self._total_tool_calls >= self.max_tool_calls:
                    output_text = (
                        "Tool call limit reached. No further tool calls will "
                        "be executed this run."
                    )
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": fc["call_id"],
                            "output": output_text,
                        }
                    )
                    executed_tool_calls.append(
                        ToolCall(
                            tool_name=fc.get("name", "?"),
                            args={},
                            result_text=output_text,
                            success=False,
                            metadata={"skipped": "tool_call_limit_reached"},
                        )
                    )
                    continue

                tool_name = fc.get("name", "")
                raw_args = fc.get("arguments", "{}")
                try:
                    args = (
                        json.loads(raw_args)
                        if isinstance(raw_args, str)
                        else (raw_args or {})
                    )
                except json.JSONDecodeError as e:
                    output_text = (
                        f"Tool call arguments could not be parsed as JSON: {e}. "
                        "Please re-issue the call with valid JSON arguments."
                    )
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": fc["call_id"],
                            "output": output_text,
                        }
                    )
                    executed_tool_calls.append(
                        ToolCall(
                            tool_name=tool_name,
                            args={"_raw": raw_args},
                            result_text=output_text,
                            success=False,
                            metadata={"error": "invalid_json_arguments"},
                        )
                    )
                    self._total_tool_calls += 1
                    continue

                if tool_name not in self.tool_map:
                    output_text = (
                        f"Unknown tool '{tool_name}'. Available tools: "
                        f"{', '.join(self.tool_names_enabled)}."
                    )
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": fc["call_id"],
                            "output": output_text,
                        }
                    )
                    executed_tool_calls.append(
                        ToolCall(
                            tool_name=tool_name,
                            args=args,
                            result_text=output_text,
                            success=False,
                            metadata={"error": "unknown_tool"},
                        )
                    )
                    self._total_tool_calls += 1
                    continue

                tool = self.tool_map[tool_name]
                tool_result = self._execute_tool(tool, args)
                self._total_tool_calls += 1

                input_items.append(
                    {
                        "type": "function_call_output",
                        "call_id": fc["call_id"],
                        "output": tool_result.output,
                    }
                )

                # Args may contain a full kernel source; truncate for the
                # trajectory record so logs don't balloon.
                logged_args = {
                    k: (
                        v[:200] + f"... [truncated, full len={len(v)}]"
                        if isinstance(v, str) and len(v) > 200
                        else v
                    )
                    for k, v in args.items()
                }
                executed_tool_calls.append(
                    ToolCall(
                        tool_name=tool_name,
                        args=logged_args,
                        result_text=tool_result.output,
                        success=tool_result.success,
                        metadata=tool_result.metadata,
                    )
                )

                # submit_kernel finalizes the run, but only if it returned a
                # real KernelExecResult. Transient errors (e.g. lock file
                # contention) leave `compiled`/`correctness` absent from
                # metadata — in those cases let the model retry.
                if tool_name == "submit_kernel":
                    submitted_kernel = args.get("kernel_code")
                    meta = tool_result.metadata or {}
                    if "compiled" in meta and "correctness" in meta:
                        is_final = True
                        try:
                            self._final_result = KernelExecResult(**meta)
                        except Exception:
                            self._final_result = None
                        break

            turn = KernelTurn(
                turn_id=turn_idx,
                messages_in=messages_in_snapshot,
                response=response_items,
                tool_calls=executed_tool_calls,
                feedback_to_model="",
                llm_latency_s=llm_latency,
                is_final=is_final,
                submitted_kernel=submitted_kernel,
            )
            trajectory.add_turn(turn)

            if is_final:
                if self.verbose:
                    print(f"[Agent] Final submission on turn {turn_idx}.")
                break

        trajectory.finish(self._final_result)
        return trajectory

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _execute_tool(self, tool: Tool, args: dict[str, Any]) -> ToolResult:
        """Execute a single tool, catching unexpected exceptions."""
        try:
            return tool.execute(self.ctx, **args)
        except Exception as e:
            return ToolResult(
                tool_name=tool.name,
                success=False,
                output=(
                    f"{tool.name} FAILED: unexpected error during tool execution.\n"
                    f"{type(e).__name__}: {e}\n"
                    f"{traceback.format_exc()}"
                ),
                metadata={"unexpected_error": str(e)},
            )
