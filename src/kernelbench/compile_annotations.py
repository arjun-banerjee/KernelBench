"""
torch.compile annotation capture for prompt construction.

Runs torch.compile in debug mode on a reference PyTorch model and extracts:
  - The generated Triton kernels
  - The ATen-level FX graph
  - Inductor fusion decisions
  - Op schedule

These artifacts are fed into the `annotated_compile` prompt option so the
model sees how torch.compile decomposed and fused the original program before
it is asked to produce a ModelNew.

Used only at prompt-construction time for the `annotated_compile` option.
Evaluation (eval_kernel_against_ref) is untouched.
"""

import multiprocessing as mp
import os
import re
import shutil
from pathlib import Path
from typing import Dict

import torch
import torch._dynamo
import torch._inductor


CAPTURE_TIMEOUT_SECONDS = 300


_LOG_PREFIX_RE = re.compile(
    r"^[VI]\d{4}\s+[\d:.]+\s+\d+\s+\S+\]\s+\[\d+/\d+\]\s+\[__\w+\]\s?",
)


def _strip_log_prefix(line: str) -> str:
    return _LOG_PREFIX_RE.sub("", line)


def _clear_inductor_caches() -> None:
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "torch", "inductor")
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
    var_tmp = Path("/var/tmp") / f"torchinductor_{os.environ.get('USER', 'user')}"
    if var_tmp.exists():
        shutil.rmtree(var_tmp, ignore_errors=True)
    for d in Path(".").glob("torch_compile_debug*"):
        shutil.rmtree(d, ignore_errors=True)


def _exec_ref_source(ref_arch_src: str) -> Dict:
    ns: Dict = {"__name__": "__kb_ref__"}
    exec(compile(ref_arch_src, "<ref_arch_src>", "exec"), ns)
    for required in ("Model", "get_init_inputs", "get_inputs"):
        if required not in ns:
            raise RuntimeError(
                f"ref_arch_src missing required symbol '{required}' "
                f"— annotated_compile needs a KernelBench-style reference."
            )
    return ns


def _run_torch_compile(ref_arch_src: str, log_path: str) -> str:
    """Compile the reference model once in debug mode and return the captured log."""
    torch._dynamo.reset()
    _clear_inductor_caches()

    # Mirror the env-setup order from the annotation experiment: delete the
    # log file and set TORCH_LOGS_OUT *before* calling set_logs, so torch
    # installs a file handler pointed at the right inode.
    os.environ["TORCHINDUCTOR_UNIQUE_KERNEL_NAMES"] = "1"
    os.environ["TORCH_COMPILE_DEBUG"] = "1"
    torch._inductor.config.fallback_random = True
    torch._inductor.config.triton.descriptive_names = "original_aten"

    if os.path.exists(log_path):
        os.remove(log_path)
    os.environ["TORCH_LOGS_OUT"] = log_path
    torch._logging.set_logs(
        output_code=True, graph_code=True, schedule=True, fusion=True
    )

    try:
        ns = _exec_ref_source(ref_arch_src)
        device = torch.device("cuda:0")
        model = ns["Model"](*ns["get_init_inputs"]()).to(device).eval()
        inputs = ns["get_inputs"]()
        inputs = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs]
        compiled = torch.compile(model, backend="inductor")
        with torch.no_grad():
            compiled(*inputs)
    finally:
        torch._logging.set_logs()
        os.environ.pop("TORCH_LOGS_OUT", None)
        os.environ.pop("TORCH_COMPILE_DEBUG", None)
        torch._dynamo.reset()
        for d in Path(".").glob("torch_compile_debug*"):
            shutil.rmtree(d, ignore_errors=True)

    return Path(log_path).read_text() if os.path.exists(log_path) else ""


def _extract_output_code(log_content: str) -> str:
    return "\n".join(
        _strip_log_prefix(line)
        for line in log_content.split("\n")
        if "[__output_code]" in line
    )


def _extract_graph(log_content: str) -> str:
    return "\n".join(
        _strip_log_prefix(line)
        for line in log_content.split("\n")
        if "[__graph_code]" in line
    )


def _extract_fusion(log_content: str, max_lines: int = 30) -> str:
    lines = []
    for raw in log_content.split("\n"):
        s = _strip_log_prefix(raw).strip()
        if s and any(kw in s.lower() for kw in ("fuse", "fusion", "fused")):
            lines.append(s)
    return "\n".join(lines[:max_lines])


def _extract_schedule(log_content: str, max_lines: int = 30) -> str:
    lines = []
    for raw in log_content.split("\n"):
        s = _strip_log_prefix(raw).strip()
        if "Topologically Sorted" in s or "Source node to ATen" in s:
            lines.append(s)
        elif s.startswith("#") and "=>" in s:
            lines.append(s)
    # dedupe while preserving order
    seen = set()
    out = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return "\n".join(out[:max_lines])


def _extract_triton_kernels(code: str) -> str:
    lines = code.split("\n")
    kernels = []
    in_kernel = False
    for line in lines:
        if "@triton.jit" in line or "@persistent_reduction" in line or "@pointwise" in line:
            in_kernel = True
        if "def triton_" in line and not in_kernel:
            in_kernel = True
        if in_kernel:
            kernels.append(line)
            if line.strip() == "''', device_str='cuda')":
                in_kernel = False
                kernels.append("")
    return "\n".join(kernels)


def _extract_call_function(code: str) -> str:
    lines = code.split("\n")
    call_lines = []
    in_call = False
    for line in lines:
        if line.startswith("def call("):
            in_call = True
        if in_call:
            call_lines.append(line)
            if line.strip().startswith("return"):
                call_lines.append("")
                in_call = False
    return "\n".join(call_lines)


def _trim_fx_graph(graph: str, max_lines: int = 50) -> str:
    sections = graph.split("TRACED GRAPH")
    final = sections[-1] if len(sections) > 1 else graph
    kept = []
    for raw in final.split("\n"):
        line = raw.strip()
        if not line or line.startswith("==="):
            continue
        if any(kw in line for kw in ("def forward", "File:", "aten.", "torch.ops", "return")):
            kept.append(line)
    return "\n".join(kept[:max_lines])


def _capture_worker(ref_arch_src: str, log_path: str, fusion_max_chars: int) -> Dict[str, str]:
    """Runs inside a spawned subprocess with its own CUDA context.

    Same pattern as KernelBench's cuda_single_eval_wrapper in
    scripts/eval_from_generations.py: fresh process so a torch.compile failure
    on the reference model cannot poison the parent process's CUDA context
    (which would break the subsequent eval step).
    """
    log_content = _run_torch_compile(ref_arch_src, log_path=log_path)
    output_code = _extract_output_code(log_content)

    triton_kernels = _extract_triton_kernels(output_code)
    call_fn = _extract_call_function(output_code)
    n_kernels = triton_kernels.count("def triton_")
    fx_graph = _trim_fx_graph(_extract_graph(log_content))
    fusion_decisions = _extract_fusion(log_content)[:fusion_max_chars]
    schedule_info = _extract_schedule(log_content)

    if not triton_kernels.strip():
        raise RuntimeError(
            "torch.compile produced no Triton kernels for this reference model — "
            "annotated_compile has nothing to show. Use a different prompt_option."
        )

    return {
        "triton_kernels": triton_kernels,
        "call_fn": call_fn,
        "fx_graph": fx_graph,
        "fusion_decisions": fusion_decisions,
        "schedule_info": schedule_info,
        "n_kernels": str(n_kernels),
    }


def build_annotated_context(
    ref_arch_src: str,
    *,
    log_path: str = "/tmp/kb_annotated_compile.log",
    fusion_max_chars: int = 2000,
    timeout: int = CAPTURE_TIMEOUT_SECONDS,
) -> Dict[str, str]:
    """Run torch.compile on the reference model and return prompt context.

    Executes the capture in an isolated subprocess (spawn context, own CUDA
    context) so a torch.compile failure on the reference cannot corrupt the
    parent process's state before the eval step. Mirrors KernelBench's
    eval_from_generations.cuda_single_eval_wrapper pattern.

    Returns a dict with keys consumed by the `annotated_compile` prompt option:
      - triton_kernels: generated Triton source (may contain multiple @triton.jit defs)
      - call_fn: the `def call(...)` driver function
      - fx_graph: trimmed ATen-level FX graph
      - fusion_decisions: Inductor fusion log lines
      - schedule_info: op schedule / source-to-ATen mapping
      - n_kernels: count of triton_* definitions (stringified for template)

    Raises RuntimeError on timeout, CUDA unavailability, or torch.compile failure.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "annotated_compile requires CUDA — torch.compile must run to extract "
            "Triton kernels and FX graph from the reference model."
        )

    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        try:
            return pool.apply_async(
                _capture_worker, args=(ref_arch_src, log_path, fusion_max_chars)
            ).get(timeout=timeout)
        except mp.TimeoutError:
            pool.terminate()
            pool.join()
            raise RuntimeError(
                f"annotated_compile capture timed out after {timeout}s on the reference model."
            )
