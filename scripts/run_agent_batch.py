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

Output layout (mirrors generate_samples.py + eval_from_generations.py)
-----------------------------------------------------------------------
runs/{run_name}/
    agent_run_config.yaml
    level_{L}_problem_{P}_trajectory.json       ← full turn history
    level_{L}_problem_{P}_sample_0_kernel.py    ← final submitted kernel (if any)
    agent_eval_results.json                      ← aggregated results (same schema as eval_results.json)

Example usage:
  uv run python scripts/run_agent_batch.py \\
    dataset_src=local level=1 \\
    run_name=my_batch_run \\
    model=gpt-5 reasoning_effort=medium \\
    num_workers=4 num_gpu_devices=4
"""

import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import pydra
import torch
from openai import OpenAI
from pydra import Config, REQUIRED
from tqdm import tqdm

from kernelbench.agent import KernelAgent, get_tools
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.utils import set_gpu_arch

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AgentBatchConfig(Config):
    def __init__(self):
        # ---- Problem ----
        self.dataset_src = REQUIRED  # "local" or "huggingface"
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.level = REQUIRED
        # Optional subset: (start_id, end_id) both inclusive, or (None, None) for all
        self.subset = (None, None)

        # ---- Model / inference ----
        self.model = REQUIRED  # e.g. "gpt-5", "gpt-5.4"
        self.openai_api_key_env = "OPENAI_API_KEY"
        self.openai_base_url = None  # None = openai.com; set for Azure / gateways
        self.reasoning_effort = None  # None | "minimal" | "low" | "medium" | "high"

        # ---- Agent loop ----
        self.max_turns = 10
        self.max_tool_calls = 30
        self.warn_turns_remaining = 2
        self.turn_delay_s = 0.0
        self.tools = "default"  # "default" | "all" | comma-separated names

        # ---- Backend / hardware ----
        self.backend = "cuda"
        self.precision = "fp32"
        self.gpu_arch = ["Ada"]
        self.timing_method = "cuda_event"
        self.num_correct_trials = 5
        self.num_perf_trials = 100

        # ---- Parallelism ----
        self.num_workers = 1
        self.num_gpu_devices = 1   # used for device assignment (worker_idx % num_gpu_devices)

        # ---- Run identity ----
        self.run_name = REQUIRED
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")

        # ---- Logging ----
        self.verbose = False
        self.save_trajectory = True

    def __repr__(self):
        return f"AgentBatchConfig({self.to_dict()})"


def _resolve_tools(tools_arg) -> list[str] | None:
    """Parse the `tools` config into the list passed to KernelAgent."""
    if isinstance(tools_arg, (list, tuple)):
        return list(tools_arg)
    tools_arg = str(tools_arg).strip().lower()
    if tools_arg == "default":
        return None  # get_tools(None) → all except profile_kernel
    if tools_arg == "all":
        from kernelbench.agent.tools import ALL_TOOLS

        return [t.name for t in ALL_TOOLS]
    return [t.strip() for t in tools_arg.split(",") if t.strip()]


def _force_backend_precision(backend: str, precision: str) -> str:
    """Apply hard-coded backend/precision constraints from the eval harness."""
    b = backend.lower()
    if b == "tilelang":
        return "fp16"
    if b == "thunderkittens":
        return "bf16"
    return precision


# ---------------------------------------------------------------------------
# Work item
# ---------------------------------------------------------------------------

@dataclass
class WorkArgs:
    problem_id: int
    device_id: int      # CUDA device index this worker should use


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
    # Bind this process to its assigned GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(work.device_id)
    device = torch.device("cuda:0")   # always 0 after CUDA_VISIBLE_DEVICES remapping

    if config.gpu_arch:
        arch = config.gpu_arch if isinstance(config.gpu_arch, list) else [config.gpu_arch]
        set_gpu_arch(arch)

    # Skip if trajectory already exists (resume support)
    traj_path = os.path.join(
        run_dir,
        f"level_{config.level}_problem_{work.problem_id}_trajectory.json",
    )
    if os.path.exists(traj_path):
        if config.verbose:
            print(f"[Worker] Skipping problem {work.problem_id}: trajectory already exists.")
        # Load and return summary from existing trajectory
        try:
            with open(traj_path) as f:
                d = json.load(f)
            return _summary_from_dict(d, config.level)
        except Exception:
            pass  # corrupted file — re-run

    try:
        # Backend precision constraints
        precision = _force_backend_precision(config.backend, config.precision)

        # API key
        api_key = os.environ.get(config.openai_api_key_env)
        if not api_key:
            print(f"[Worker] ERROR: env var '{config.openai_api_key_env}' is not set.")
            return None

        # OpenAI client
        client_kwargs: dict = {"api_key": api_key}
        if config.openai_base_url:
            client_kwargs["base_url"] = config.openai_base_url
        client = OpenAI(**client_kwargs)

        # Load problem
        dataset = construct_kernelbench_dataset(
            level=config.level,
            source=config.dataset_src,
            dataset_name=config.dataset_name,
        )
        problem = dataset.get_problem_by_id(work.problem_id)
        ref_arch_src = problem.code
        problem_name = problem.name

        # Resolve tools
        tool_names = _resolve_tools(config.tools)
        tools = get_tools(tool_names)

        # Build cache dir
        build_dir = os.path.join(
            run_dir, f"level_{config.level}_problem_{work.problem_id}_cache"
        )
        os.makedirs(build_dir, exist_ok=True)

        # Run agent
        agent = KernelAgent(
            problem_id=work.problem_id,
            level=config.level,
            problem_name=problem_name,
            ref_arch_src=ref_arch_src,
            client=client,
            model=config.model,
            run_name=config.run_name,
            tool_names=[t.name for t in tools],
            max_turns=config.max_turns,
            max_tool_calls=config.max_tool_calls,
            backend=config.backend,
            precision=precision,
            device=device,
            build_dir=build_dir,
            num_correct_trials=config.num_correct_trials,
            num_perf_trials=config.num_perf_trials,
            timing_method=config.timing_method,
            reasoning_effort=config.reasoning_effort,
            warn_turns_remaining=config.warn_turns_remaining,
            turn_delay_s=float(config.turn_delay_s),
            verbose=config.verbose,
        )

        trajectory = agent.run()

        # Save trajectory JSON
        if config.save_trajectory:
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


def _check_trajectory_exists(run_dir: str, level: int, problem_id: int) -> bool:
    path = os.path.join(run_dir, f"level_{level}_problem_{problem_id}_trajectory.json")
    return os.path.exists(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@pydra.main(base=AgentBatchConfig)
def main(config: AgentBatchConfig):
    # Coerce string "True"/"False" from the CLI to actual bools.
    for bool_field in ("verbose", "save_trajectory"):
        val = getattr(config, bool_field)
        if isinstance(val, str):
            setattr(config, bool_field, val.lower() in ("true", "1", "yes"))

    print(f"[run_agent_batch] Config: {config}")

    # API key check
    api_key = os.environ.get(config.openai_api_key_env)
    if not api_key:
        sys.exit(
            f"[run_agent_batch] FATAL: env var '{config.openai_api_key_env}' is not set. "
            f"Either export it or pass openai_api_key_env=<other_var_name>."
        )

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
            work_items.append(WorkArgs(
                problem_id=int(pid),
                device_id=i % config.num_gpu_devices,
            ))

    total = len(problem_ids)
    print(f"[run_agent_batch] Level {config.level}: {total} problems total")
    print(f"  Already completed: {already_done}")
    print(f"  To run:           {len(work_items)}")
    print(f"  Workers:          {config.num_workers}  (GPU devices: {config.num_gpu_devices})")
    print(f"  Tools:            {_resolve_tools(config.tools) or 'default (no profiling)'}")
    print(f"  Max turns:        {config.max_turns}  |  Max tool calls: {config.max_tool_calls}")

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
                        print(f"[run_agent_batch] Worker failed for problem {work.problem_id}: {e}")

    elapsed = time.time() - t_start
    n_correct = sum(1 for r in results if r.get("correctness"))
    n_compiled = sum(1 for r in results if r.get("compiled"))
    print(f"\n{'='*60}")
    print(f"[run_agent_batch] Done in {elapsed:.1f}s")
    print(f"  Attempted: {len(work_items)}  |  Results returned: {len(results)}")
    print(f"  Compiled:  {n_compiled}/{len(results)}")
    print(f"  Correct:   {n_correct}/{len(results)}")
    print(f"{'='*60}")

    # Aggregate all results (including previously completed ones)
    _aggregate_results(run_dir, config.level, problem_ids)


def _aggregate_results(run_dir: str, level: int, problem_ids):
    """
    Collect all per-problem summaries into agent_eval_results.json.
    Schema matches eval_from_generations.py's eval_results.json for compatibility.
    """
    all_results = {}
    for pid in problem_ids:
        traj_path = os.path.join(run_dir, f"level_{level}_problem_{pid}_trajectory.json")
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
    print(f"  Compiled:           {n_compiled}/{n}  ({100*n_compiled/n:.1f}%)" if n else "")
    print(f"  Correct:            {n_correct}/{n}  ({100*n_correct/n:.1f}%)" if n else "")
    print(f"  Avg turns used:     {avg_turns:.1f}")


if __name__ == "__main__":
    main()