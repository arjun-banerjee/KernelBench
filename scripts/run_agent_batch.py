"""
run_agent_batch.py — Batch multi-turn agent over all problems in a level.

Parallelism model
-----------------
The agent loop interleaves LLM API calls (network I/O) with GPU eval tool calls.
Because GPU contexts cannot safely be shared across threads, each worker is a
separate *process* with an assigned CUDA device.

  num_workers = number of parallel agent processes
               (set this to num_gpu_devices — one process per GPU)

For a single GPU, set num_workers=1 (agents run sequentially, GPU is reused).
For N GPUs, set num_workers=N — problems are distributed round-robin.

The vLLM / local inference server is shared; all workers call it concurrently.

Output layout (mirrors generate_samples.py + eval_from_generations.py)
-----------------------------------------------------------------------
runs/{run_name}/
    agent_run_config.yaml
    level_{L}_problem_{P}_trajectory.json       ← full turn history
    level_{L}_problem_{P}_sample_0_kernel.py    ← final submitted kernel (if any)
    agent_eval_results.json                      ← aggregated results (same schema as eval_results.json)

Example usage:
  uv run python scripts/run_agent_batch.py \\
    dataset_src=local level=5 \\
    run_name=test_dist_agent \\
    runs_dir=/scratch/abaner/my_kernel_runs/qwen3-coder-agent \\
    num_workers=1 \\
    server_type=local server_address=localhost server_port=8000 \\
    model_name=Qwen/Qwen2.5-Coder-32B-Instruct \\
    temperature=0 max_tokens=8192 \\
    max_turns=8 max_tool_calls=25 \\
    tools=compile_kernel,run_correctness,static_check,submit_kernel

To add nsight profiling (requires ncu + permissions):
    tools=compile_kernel,run_correctness,profile_kernel,get_gpu_specs,static_check,submit_kernel
"""

import json
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Optional

import pydra
import torch
from pydra import Config, REQUIRED
from tqdm import tqdm

from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.utils import create_inference_server_from_presets, set_gpu_arch
from kernelbench.agent import KernelAgent, get_tools

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AgentBatchConfig(Config):
    def __init__(self):
        # Dataset
        self.dataset_src = REQUIRED
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.level = REQUIRED
        # Optional subset: (start_id, end_id) both inclusive, or (None, None) for all
        self.subset = (None, None)

        # Run identity
        self.run_name = REQUIRED
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")

        # Parallelism: set to number of GPU devices
        self.num_workers = 1
        self.num_gpu_devices = (
            1  # used for device assignment (worker_idx % num_gpu_devices)
        )

        # Inference
        self.server_type = REQUIRED
        self.model_name = REQUIRED
        self.max_tokens = None
        self.temperature = 0.0
        self.server_address = None
        self.server_port = None
        self.is_reasoning_model = False
        self.reasoning_effort = None
        self.budget_tokens = 0

        # Agent loop
        self.max_turns = 8
        self.max_tool_calls = 25
        self.warn_turns_remaining = 2
        # Minimum seconds to sleep between turns (helps with tight per-minute
        # output-token quotas, e.g. Anthropic free-tier: 4k tokens/min).
        self.turn_delay_s = 0.0
        # Tool call XML dialect shown to the model in the system prompt.
        # "canonical" (default) | "nemotron" | "auto"
        # Use "nemotron" for Nemotron / Llama function-calling models.
        self.tool_format = "canonical"
        # "default" | "all" | comma-separated list
        self.tools = "default"

        # Backend / hardware
        self.backend = "cuda"
        self.precision = "fp32"
        self.gpu_arch = ["Hopper"]
        self.include_hardware_info = False
        self.timing_method = "cuda_event"
        self.num_correct_trials = 5
        self.num_perf_trials = 100

        # Logging
        self.verbose = False
        self.log_prompt = False  # save initial problem prompt to file

    def __repr__(self):
        return f"AgentBatchConfig({self.to_dict()})"


# ---------------------------------------------------------------------------
# Work item
# ---------------------------------------------------------------------------


@dataclass
class WorkArgs:
    problem_id: int
    device_id: int  # CUDA device index this worker should use


# ---------------------------------------------------------------------------
# Worker function (runs in a child process)
# ---------------------------------------------------------------------------


def run_agent_worker(
    work: WorkArgs,
    config: AgentBatchConfig,
    run_dir: str,
) -> Optional[dict]:
    """
    Run one KernelAgent on one problem in a subprocess.
    Returns a dict summary suitable for aggregation, or None on failure.
    """
    import os
    import torch

    # Bind this process to its assigned GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(work.device_id)
    device = torch.device("cuda:0")  # always 0 after CUDA_VISIBLE_DEVICES remapping

    if config.gpu_arch:
        arch = (
            config.gpu_arch if isinstance(config.gpu_arch, list) else [config.gpu_arch]
        )
        set_gpu_arch(arch)

    # Skip if trajectory already exists (resume support)
    traj_path = os.path.join(
        run_dir,
        f"level_{config.level}_problem_{work.problem_id}_trajectory.json",
    )
    if os.path.exists(traj_path):
        if config.verbose:
            print(
                f"[Worker] Skipping problem {work.problem_id}: trajectory already exists."
            )
        # Load and return summary from existing trajectory
        try:
            with open(traj_path) as f:
                d = json.load(f)
            return _summary_from_dict(d, config.level)
        except Exception:
            pass  # corrupted file — re-run

    try:
        # Load problem
        dataset = construct_kernelbench_dataset(
            level=config.level,
            source=config.dataset_src,
            dataset_name=config.dataset_name,
        )
        problem = dataset.get_problem_by_id(work.problem_id)
        ref_arch_src = problem.code
        problem_name = problem.name

        # Save prompt if requested
        if config.log_prompt:
            from kernelbench.prompt_constructor_toml import get_prompt_for_backend

            initial_prompt = get_prompt_for_backend(
                ref_arch_src,
                config.backend,
                option="one_shot",
                precision=config.precision,
            )
            prompt_path = os.path.join(
                run_dir,
                f"level_{config.level}_problem_{work.problem_id}_sample_0_prompt.txt",
            )
            with open(prompt_path, "w") as f:
                f.write(initial_prompt)

        # Build inference function
        inference_fn = create_inference_server_from_presets(
            server_type=config.server_type,
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            verbose=False,
            time_generation=False,
            is_reasoning_model=config.is_reasoning_model,
            reasoning_effort=config.reasoning_effort,
            budget_tokens=config.budget_tokens,
            server_address=config.server_address,
            server_port=config.server_port,
        )
        inference_fn.__name__ = config.model_name

        # Resolve tools
        tool_names = _resolve_tools(config.tools)
        tools = get_tools(tool_names)

        # Build cache dir
        build_dir = os.path.join(
            run_dir, f"level_{config.level}_problem_{work.problem_id}_cache"
        )
        os.makedirs(build_dir, exist_ok=True)

        # Backend precision constraints
        precision = config.precision
        backend = config.backend.lower()
        if backend == "tilelang":
            precision = "fp16"
        if backend == "thunderkittens":
            precision = "bf16"

        # Run agent
        agent = KernelAgent(
            problem_id=work.problem_id,
            level=config.level,
            problem_name=problem_name,
            ref_arch_src=ref_arch_src,
            inference_fn=inference_fn,
            run_name=config.run_name,
            tool_names=[t.name for t in tools],
            max_turns=config.max_turns,
            max_tool_calls=config.max_tool_calls,
            backend=backend,
            precision=precision,
            device=device,
            build_dir=build_dir,
            num_correct_trials=config.num_correct_trials,
            num_perf_trials=config.num_perf_trials,
            timing_method=config.timing_method,
            include_hardware_info=config.include_hardware_info,
            warn_turns_remaining=config.warn_turns_remaining,
            tool_format=config.tool_format,
            turn_delay_s=float(config.turn_delay_s),
            verbose=config.verbose,
        )

        trajectory = agent.run()

        # Save trajectory JSON
        trajectory.save(traj_path)

        # Also save the final submitted kernel as a .py file for compatibility
        # with eval_from_generations.py (in case someone wants to re-evaluate)
        submitted_kernel = _extract_final_kernel(trajectory)
        if submitted_kernel:
            kernel_path = os.path.join(
                run_dir,
                f"level_{config.level}_problem_{work.problem_id}_sample_0_kernel.py",
            )
            with open(kernel_path, "w") as f:
                f.write(submitted_kernel)

        return _summary_from_dict(trajectory.to_dict(), config.level)

    except Exception as e:
        import traceback

        print(f"[Worker] ERROR on problem {work.problem_id}: {e}")
        traceback.print_exc()
        return None


def _extract_final_kernel(trajectory) -> Optional[str]:
    """Pull the last submitted kernel code from the trajectory."""
    for turn in reversed(trajectory.turns):
        if turn.submitted_kernel:
            return turn.submitted_kernel
    return None


def _summary_from_dict(d: dict, level: int) -> dict:
    """Build the per-problem summary dict for aggregation."""
    fr = d.get("final_result") or {}
    return {
        "problem_id": d.get("problem_id"),
        "level": level,
        "problem_name": d.get("problem_name"),
        "outcome": d.get("outcome"),
        "total_turns": d.get("total_turns"),
        "total_tool_calls": d.get("total_tool_calls"),
        "compiled": fr.get("compiled", False),
        "correctness": fr.get("correctness", False),
        "runtime": fr.get("runtime", -1.0),
        "runtime_stats": fr.get("runtime_stats", {}),
        "ref_runtime": fr.get("ref_runtime", -1.0),
        "metadata": fr.get("metadata", {}),
    }


def _resolve_tools(tools_arg) -> Optional[list]:
    if isinstance(tools_arg, (list, tuple)):
        return list(tools_arg)
    tools_arg = str(tools_arg).strip().lower()
    if tools_arg == "default":
        return None
    if tools_arg == "all":
        from kernelbench.agent.tools import ALL_TOOLS

        return [t.name for t in ALL_TOOLS]
    return [t.strip() for t in tools_arg.split(",") if t.strip()]


def _check_trajectory_exists(run_dir: str, level: int, problem_id: int) -> bool:
    path = os.path.join(run_dir, f"level_{level}_problem_{problem_id}_trajectory.json")
    return os.path.exists(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@pydra.main(base=AgentBatchConfig)
def main(config: AgentBatchConfig):
    from kernelbench.utils import SERVER_PRESETS

    # Resolve presets
    if config.server_type and config.server_type in SERVER_PRESETS:
        preset = SERVER_PRESETS[config.server_type]
        if not config.model_name or config.model_name == "None":
            config.model_name = preset.get("model_name")
        if not config.max_tokens or config.max_tokens == "None":
            config.max_tokens = preset.get("max_tokens", 8192)
        if config.temperature is None or config.temperature == "None":
            config.temperature = preset.get("temperature", 0.7)

    for bool_field in (
        "is_reasoning_model",
        "include_hardware_info",
        "verbose",
        "log_prompt",
    ):
        val = getattr(config, bool_field)
        if isinstance(val, str):
            setattr(config, bool_field, val.lower() in ["true", "1", "yes"])

    print(f"[run_agent_batch] Config: {config}")

    # Dataset
    dataset = construct_kernelbench_dataset(
        level=config.level,
        source=config.dataset_src,
        dataset_name=config.dataset_name,
    )
    all_problem_ids = dataset.get_problem_ids()

    # Apply subset filter
    if config.subset == (None, None):
        problem_ids = all_problem_ids
    else:
        start, end = config.subset
        problem_ids = [p for p in all_problem_ids if start <= p <= end]

    # Run directory
    run_dir = os.path.join(config.runs_dir, config.run_name)
    run_exists = os.path.exists(run_dir)
    if run_exists:
        print(f"\n⚠️  WARNING: Run directory already exists: {run_dir}")
        print(f"   Existing trajectories will be skipped.\n")
    os.makedirs(run_dir, exist_ok=True)
    pydra.save_yaml(config.to_dict(), os.path.join(run_dir, "agent_run_config.yaml"))

    # Build work list, assigning GPU devices round-robin
    work_items = []
    already_done = 0
    for i, pid in enumerate(problem_ids):
        if _check_trajectory_exists(run_dir, config.level, pid):
            already_done += 1
        else:
            work_items.append(
                WorkArgs(
                    problem_id=int(pid),
                    device_id=i % config.num_gpu_devices,
                )
            )

    total = len(problem_ids)
    print(f"[run_agent_batch] Level {config.level}: {total} problems total")
    print(f"  Already completed: {already_done}")
    print(f"  To run:           {len(work_items)}")
    print(
        f"  Workers:          {config.num_workers}  (GPU devices: {config.num_gpu_devices})"
    )
    print(
        f"  Tools:            {_resolve_tools(config.tools) or 'default (no profiling)'}"
    )
    print(
        f"  Max turns:        {config.max_turns}  |  Max tool calls: {config.max_tool_calls}"
    )

    if not work_items:
        print(f"\n✅ All {total} trajectories already exist in {run_dir}")
        _aggregate_results(run_dir, config.level, problem_ids)
        return

    # Spawn processes (required for CUDA)
    mp.set_start_method("spawn", force=True)

    results = []
    t_start = time.time()

    if config.num_workers == 1:
        # Sequential — simpler, avoids multiprocessing overhead
        for work in tqdm(work_items, desc="Agent runs"):
            result = run_agent_worker(work, config, run_dir)
            if result is not None:
                results.append(result)
    else:
        # Parallel across GPUs
        with tqdm(total=len(work_items), desc="Agent runs") as pbar:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
                futures = {
                    executor.submit(run_agent_worker, work, config, run_dir): work
                    for work in work_items
                }
                for future in as_completed(futures):
                    pbar.update(1)
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        work = futures[future]
                        print(
                            f"[run_agent_batch] Worker failed for problem {work.problem_id}: {e}"
                        )

    elapsed = time.time() - t_start
    n_correct = sum(1 for r in results if r.get("correctness"))
    n_compiled = sum(1 for r in results if r.get("compiled"))
    print(f"\n{'=' * 60}")
    print(f"[run_agent_batch] Done in {elapsed:.1f}s")
    print(f"  Attempted: {len(work_items)}  |  Results returned: {len(results)}")
    print(f"  Compiled:  {n_compiled}/{len(results)}")
    print(f"  Correct:   {n_correct}/{len(results)}")
    print(f"{'=' * 60}")

    # Aggregate all results (including previously completed ones)
    _aggregate_results(run_dir, config.level, problem_ids)


def _aggregate_results(run_dir: str, level: int, problem_ids):
    """
    Collect all per-problem summaries into agent_eval_results.json.
    Schema matches eval_from_generations.py's eval_results.json for compatibility.
    """
    all_results = {}
    for pid in problem_ids:
        traj_path = os.path.join(
            run_dir, f"level_{level}_problem_{pid}_trajectory.json"
        )
        if not os.path.exists(traj_path):
            continue
        try:
            with open(traj_path) as f:
                d = json.load(f)
            all_results[str(pid)] = _summary_from_dict(d, level)
        except Exception as e:
            print(f"[Aggregate] Could not read trajectory for problem {pid}: {e}")

    out_path = os.path.join(run_dir, "agent_eval_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Quick stats
    n = len(all_results)
    n_correct = sum(1 for r in all_results.values() if r.get("correctness"))
    n_compiled = sum(1 for r in all_results.values() if r.get("compiled"))
    avg_turns = (
        sum(r.get("total_turns", 0) for r in all_results.values()) / n if n else 0
    )
    print(f"\n[Aggregate] Results written to: {out_path}")
    print(f"  Problems evaluated: {n}")
    print(
        f"  Compiled:           {n_compiled}/{n}  ({100 * n_compiled / n:.1f}%)"
        if n
        else ""
    )
    print(
        f"  Correct:            {n_correct}/{n}  ({100 * n_correct / n:.1f}%)"
        if n
        else ""
    )
    print(f"  Avg turns used:     {avg_turns:.1f}")


if __name__ == "__main__":
    main()
