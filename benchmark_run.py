"""Command-line runner for reproducible baseline MAPF-LNS benchmarks."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

from benchmark import (
    make_bottleneck,
    make_corridor_swap,
    make_dense_empty,
    make_empty,
    make_hub,
    make_random_obstacles,
)
from mapf_lns import LNSConfig, PlanningFailure, prioritized_planning, run_lns
from mapf_lns.destroy import _neighborhood_size


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        choices=("empty", "random", "corridor", "dense", "bottleneck", "hub"),
        required=True,
    )
    parser.add_argument("--size", type=int, default=16)
    parser.add_argument("--agents", type=int, default=20)
    parser.add_argument("--obstacle-ratio", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.3)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=None)
    return parser


def _mean(values: list[int]) -> float | None:
    return statistics.fmean(values) if values else None


def _print_table(result: dict[str, Any]) -> None:
    width = max(len(key) for key in result)
    print(f"{'metric':<{width}} | value")
    print(f"{'-' * width}-+----------------")
    for key, value in result.items():
        print(f"{key:<{width}} | {value}")


def main() -> int:
    args = _parser().parse_args()
    if args.scenario == "empty":
        instance = make_empty(args.size, args.agents, args.seed)
    elif args.scenario == "random":
        instance = make_random_obstacles(
            args.size, args.agents, args.obstacle_ratio, args.seed
        )
    elif args.scenario == "corridor":
        instance = make_corridor_swap(args.agents, args.seed)
    elif args.scenario == "dense":
        instance = make_dense_empty(args.size, args.agents, args.seed)
    elif args.scenario == "bottleneck":
        instance = make_bottleneck(args.size, args.agents, args.seed)
    else:
        instance = make_hub(args.size, args.agents, args.seed)

    pp_started = time.perf_counter()
    try:
        initial = prioritized_planning(instance)
    except PlanningFailure:
        pp_runtime = time.perf_counter() - pp_started
        result: dict[str, Any] = {
            "scenario": args.scenario,
            "size": None if args.scenario == "corridor" else args.size,
            "agents": args.agents,
            "gamma": args.gamma,
            "iterations_requested": args.iterations,
            "initial_soc": "PP_FAILED",
            "best_soc": None,
            "improvement": None,
            "improvement_ratio": None,
            "iterations_run": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "repair_failed_count": 0,
            "accept_rate": None,
            "mean_neighborhood_size": _neighborhood_size(args.agents, args.gamma),
            "mean_candidate_soc_when_accepted": None,
            "mean_candidate_soc_when_rejected": None,
            "runtime_seconds": pp_runtime,
        }
        print("PP_FAILED")
        _print_table(result)
        print("JSON: " + json.dumps(result, sort_keys=True))
        return 0

    initial_soc = initial.compute_soc()
    config = LNSConfig(
        max_iterations=args.iterations,
        gamma=args.gamma,
        seed=args.seed,
        time_limit_seconds=args.time_limit,
    )
    started = time.perf_counter()
    best, logs = run_lns(instance, config)
    runtime = time.perf_counter() - started

    accepted = [entry for entry in logs if entry.accepted]
    rejected = [entry for entry in logs if not entry.accepted]
    repair_failed = [entry for entry in logs if entry.soc_candidate is None]
    accepted_socs = [
        entry.soc_candidate for entry in accepted if entry.soc_candidate is not None
    ]
    rejected_socs = [
        entry.soc_candidate for entry in rejected if entry.soc_candidate is not None
    ]
    best_soc = best.compute_soc()
    improvement = initial_soc - best_soc
    result = {
        "scenario": args.scenario,
        "size": None if args.scenario == "corridor" else args.size,
        "agents": args.agents,
        "gamma": args.gamma,
        "iterations_requested": args.iterations,
        "initial_soc": initial_soc,
        "best_soc": best_soc,
        "improvement": improvement,
        "improvement_ratio": improvement / initial_soc if initial_soc else 0.0,
        "iterations_run": len(logs),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "repair_failed_count": len(repair_failed),
        "accept_rate": len(accepted) / len(logs) if logs else 0.0,
        "mean_neighborhood_size": _neighborhood_size(args.agents, args.gamma),
        "mean_candidate_soc_when_accepted": _mean(accepted_socs),
        "mean_candidate_soc_when_rejected": _mean(rejected_socs),
        "runtime_seconds": runtime,
    }
    _print_table(result)
    print("JSON: " + json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
