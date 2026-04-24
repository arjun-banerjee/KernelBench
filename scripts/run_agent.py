"""
run_agent.py — single-problem multi-turn agent entry point.

Runs one KernelAgent against one (level, problem_id) pair using the OpenAI
Responses API (or any compatible endpoint, e.g. Azure OpenAI, via base_url).

Example usage (OpenAI direct):
  uv run python scripts/run_agent.py \\
    dataset_src=local level=1 problem_id=1 \\
    model=gpt-5 reasoning_effort=medium \\
    backend=cuda precision=fp32 \\
    max_turns=10 max_tool_calls=30 \\
    run_name=my_agent_run

Example usage (Azure OpenAI):
  uv run python scripts/run_agent.py \\
    dataset_src=local level=1 problem_id=1 \\
    model=gpt-5.4 \\
    openai_base_url=https://thava-openai.cognitiveservices.azure.com/openai/v1/ \\
    openai_api_key_env=AZURE_OPENAI_API_KEY \\
    backend=cuda precision=fp32 \\
    run_name=my_agent_run

Tool selection:
  tools=default                                                     # all except profile_kernel
  tools=all                                                         # everything (requires ncu)
  tools=compile_kernel,run_correctness,static_check,submit_kernel   # explicit list
"""

import os
import sys

import pydra
import torch
from openai import OpenAI
from pydra import Config, REQUIRED

from kernelbench.agent import KernelAgent, get_tools
from kernelbench.dataset import construct_kernelbench_dataset
from kernelbench.utils import set_gpu_arch

REPO_TOP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class AgentConfig(Config):
    def __init__(self):
        # ---- Problem ----
        self.dataset_src = REQUIRED  # "local" or "huggingface"
        self.dataset_name = "ScalingIntelligence/KernelBench"
        self.level = REQUIRED
        self.problem_id = REQUIRED

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

        # ---- Run identity ----
        self.run_name = "agent_run"
        self.runs_dir = os.path.join(REPO_TOP_DIR, "runs")

        # ---- Logging ----
        self.verbose = False
        self.save_trajectory = True

    def __repr__(self):
        return f"AgentConfig({self.to_dict()})"


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


@pydra.main(base=AgentConfig)
def main(config: AgentConfig):
    # Coerce string "True"/"False" from the CLI to actual bools.
    for bool_field in ("verbose", "save_trajectory"):
        val = getattr(config, bool_field)
        if isinstance(val, str):
            setattr(config, bool_field, val.lower() in ("true", "1", "yes"))

    print(f"[run_agent] Config: {config}")

    # ---- GPU arch ----
    if config.gpu_arch:
        if not isinstance(config.gpu_arch, list):
            config.gpu_arch = [config.gpu_arch]
        set_gpu_arch(config.gpu_arch)

    # ---- Backend precision constraints ----
    config.precision = _force_backend_precision(config.backend, config.precision)

    # ---- API key ----
    api_key = os.environ.get(config.openai_api_key_env)
    if not api_key:
        sys.exit(
            f"[run_agent] FATAL: env var '{config.openai_api_key_env}' is not set. "
            f"Either export it or pass openai_api_key_env=<other_var_name>."
        )

    # ---- OpenAI client ----
    client_kwargs: dict = {"api_key": api_key}
    if config.openai_base_url:
        client_kwargs["base_url"] = config.openai_base_url
    client = OpenAI(**client_kwargs)

    # ---- Load problem ----
    dataset = construct_kernelbench_dataset(
        level=config.level,
        source=config.dataset_src,
        dataset_name=config.dataset_name,
    )
    problem = dataset.get_problem_by_id(config.problem_id)
    ref_arch_src = problem.code
    problem_name = problem.name
    print(
        f"[run_agent] Problem: level={config.level} id={config.problem_id} "
        f"name={problem_name}"
    )

    # ---- Tool selection ----
    tool_names = _resolve_tools(config.tools)
    tools = get_tools(tool_names)
    print(f"[run_agent] Tools enabled: {[t.name for t in tools]}")

    # ---- Device + build cache ----
    device = torch.device(
        f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    )
    build_dir = os.path.join(
        config.runs_dir,
        config.run_name,
        f"level_{config.level}_problem_{config.problem_id}_cache",
    )
    os.makedirs(build_dir, exist_ok=True)

    # ---- Agent ----
    agent = KernelAgent(
        problem_id=config.problem_id,
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
        precision=config.precision,
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

    print(
        f"[run_agent] Starting loop "
        f"(max_turns={config.max_turns}, max_tool_calls={config.max_tool_calls})"
    )
    trajectory = agent.run()

    # ---- Summary ----
    print()
    print("=" * 60)
    print("[run_agent] Agent run complete.")
    print(f"  Outcome:           {trajectory.outcome}")
    print(f"  Total turns:       {trajectory.total_turns}")
    print(f"  Total tool calls:  {trajectory.total_tool_calls}")
    if trajectory.final_result:
        r = trajectory.final_result
        print(f"  Compiled:          {r.compiled}")
        print(f"  Correct:           {r.correctness}")
        if r.runtime > 0:
            print(f"  Kernel runtime:    {r.runtime:.2f} μs")
    print("=" * 60)

    # ---- Save trajectory ----
    if config.save_trajectory:
        traj_path = os.path.join(
            config.runs_dir,
            config.run_name,
            f"level_{config.level}_problem_{config.problem_id}_trajectory.json",
        )
        trajectory.save(traj_path)
        print(f"[run_agent] Trajectory saved to: {traj_path}")


if __name__ == "__main__":
    main()
