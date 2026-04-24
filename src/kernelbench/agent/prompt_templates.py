"""
Prompts for the KernelBench multi-turn agent.

Three public builders:
    build_system_prompt(max_turns, max_tool_calls, backend) -> str
        The system prompt, passed as `instructions` to the Responses API.
        Stable across turns within a run.

    build_problem_message(ref_arch_src, backend, precision) -> str
        The first user-role message: task statement, backend-specific output
        format, and the reference PyTorch source.

    build_turn_warning_message(turns_remaining, tool_calls_remaining) -> str
        Injected as a user-role message every turn once the warning threshold
        is crossed.

Tool descriptions live in `tools.py` as JSON-schema `description` fields and
are not duplicated here.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Backend display-name map (shared between system prompt and problem message)
# ---------------------------------------------------------------------------

_BACKEND_DISPLAY: dict[str, str] = {
    "cuda": "CUDA",
    "triton": "Triton",
    "tilelang": "TileLang",
    "cute": "CUTLASS/CuTe",
}


def _backend_display(backend: str) -> str:
    """Return a human-readable backend name. Raises for unsupported backends."""
    key = backend.lower()
    if key not in _BACKEND_DISPLAY:
        raise NotImplementedError(
            f"Backend '{backend}' is not supported by the agent prompts. "
            f"Supported backends: {sorted(_BACKEND_DISPLAY.keys())}."
        )
    return _BACKEND_DISPLAY[key]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert GPU kernel engineer. Your task is to write a high-performance \
custom {backend_display} kernel that replaces a PyTorch reference implementation, \
producing numerically identical results.

You work iteratively: call tools to compile, test, and profile your kernel, read \
the results, and refine.

## Session limits

- You have {max_turns} turns. One turn is one response from you, which may \
contain reasoning and zero or more tool calls.
- You have {max_tool_calls} tool calls total across all turns.
- Only `submit_kernel` records your final result. It ends the session.

## Correctness requirements

- Outputs must match the reference within tolerance: fp32 uses atol=rtol=1e-4; \
fp16/bf16 use atol=rtol=1e-2.
- Do not use try/except fallbacks to the reference, patch timing functions, or \
otherwise work around correctness or timing measurement — these are detected \
and cause evaluation failure.
"""


def build_system_prompt(
    *,
    max_turns: int,
    max_tool_calls: int,
    backend: str,
) -> str:
    """Build the agent's system prompt (the API's `instructions` parameter)."""
    return _SYSTEM_PROMPT_TEMPLATE.format(
        backend_display=_backend_display(backend),
        max_turns=max_turns,
        max_tool_calls=max_tool_calls,
    )


# ---------------------------------------------------------------------------
# Per-backend output-format blocks
# ---------------------------------------------------------------------------


def _output_format_cuda() -> str:
    return """\
## Output format

Your submission is a complete Python file that defines a class called \
`ModelNew`. The file is executed with `exec()` before we instantiate \
`ModelNew`, so any module-level setup runs first.

Use `torch.utils.cpp_extension.load_inline` to compile and bind CUDA source at \
module load time, then call the compiled extension from `ModelNew.forward`. Do \
not submit raw CUDA C or a standalone .cu file."""


def _output_format_triton() -> str:
    return """\
## Output format

Your submission is a complete Python file that defines a class called \
`ModelNew`. Write the kernel as a function decorated with `@triton.jit` (or \
`@triton.autotune`) using `triton.language` (commonly aliased as `tl`). Launch \
the kernel from `ModelNew.forward` with an appropriate grid."""


def _output_format_tilelang() -> str:
    return """\
## Output format

Your submission is a complete Python file that defines a class called \
`ModelNew`. Write the kernel as a `@T.prim_func` using `tilelang.language` \
(aliased as `T`), compile it with `tilelang.compile(..., target="cuda")`, and \
invoke the compiled kernel from `ModelNew.forward`. Note: TileLang requires \
fp16 or bf16 precision."""


def _output_format_cute() -> str:
    return """\
## Output format

Your submission is a complete Python file that defines a class called \
`ModelNew`. Use the CUTLASS/CuTe Python bindings (`cutlass`, and the `cute::` \
namespace in any inlined C++) to build the kernel, and invoke it from \
`ModelNew.forward`."""


_OUTPUT_FORMAT_BUILDERS = {
    "cuda": _output_format_cuda,
    "triton": _output_format_triton,
    "tilelang": _output_format_tilelang,
    "cute": _output_format_cute,
}


def _output_format(backend: str) -> str:
    """Return the output-format block for the given backend, or raise."""
    key = backend.lower()
    builder = _OUTPUT_FORMAT_BUILDERS.get(key)
    if builder is None:
        raise NotImplementedError(
            f"No output-format prompt for backend '{backend}'. "
            f"Supported backends: {sorted(_OUTPUT_FORMAT_BUILDERS.keys())}."
        )
    return builder()


# ---------------------------------------------------------------------------
# Problem message (first user turn)
# ---------------------------------------------------------------------------

_PROBLEM_TEMPLATE = """\
## Task

Optimize the following PyTorch model by replacing its forward computation with \
a custom {backend_display} kernel. Your `ModelNew` class must:

1. Accept the same constructor arguments as `Model`.
2. Implement `forward()` with the same signature.
3. Produce numerically equivalent outputs at {precision} precision.
4. Be faster than the PyTorch reference.

{output_format}

## Reference implementation

```python
{ref_arch_src}
```
"""


def build_problem_message(
    *,
    ref_arch_src: str,
    backend: str,
    precision: str,
) -> str:
    """Build the first user-role message describing the problem."""
    return _PROBLEM_TEMPLATE.format(
        backend_display=_backend_display(backend),
        precision=precision,
        output_format=_output_format(backend),
        ref_arch_src=ref_arch_src.rstrip(),
    )


# ---------------------------------------------------------------------------
# Turn-limit warning
# ---------------------------------------------------------------------------


def build_turn_warning_message(
    turns_remaining: int,
    tool_calls_remaining: int,
) -> str:
    """Build the soft warning injected as a user message near the session cap."""
    turn_word = "turn" if turns_remaining == 1 else "turns"
    call_word = "tool call" if tool_calls_remaining == 1 else "tool calls"
    return (
        f"Session warning: {turns_remaining} {turn_word} and "
        f"{tool_calls_remaining} {call_word} remain. "
        "Submit your best kernel soon."
    )
