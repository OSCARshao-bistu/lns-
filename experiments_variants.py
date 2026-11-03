"""Run the five fixed N=32 LNS variants on the local Moving AI benchmark.

Runtime includes initialization. The 60-second LNS budget starts after the
initializer, preserving run_lns semantics. EECBS and PPS use their own defaults.
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys
import time

from mapf_lns.conflicts import find_all_conflicts
from mapf_lns.lns import LNSConfig, run_lns
from mapf_lns.models import Agent, MAPFInstance, Solution
from mapf_lns.prioritized_planning import PlanningFailure


ROOT = Path(__file__).resolve().parent
NUM_AGENTS = 32
GAMMA = 0.3
MAX_ITERATIONS = 1000
TIME_LIMIT_SECONDS = 60.0
SEED = 0
VARIANTS = (
    ("Base-ALNS", "pp", "alns"),
    ("EECBS-ALNS", "eecbs", "alns"),
    ("PPS-Random", "pps", "random"),
    ("Agent-based", "pp", "agent_based"),
    ("Map-based", "pp", "map_based"),
)
FIELDNAMES = (
    "variant", "init_method", "neighborhood_mode", "initial_soc", "final_soc",
    "runtime_s", "status", "iterations", "error",
)


def load_instance(map_path: Path, scen_path: Path, num_agents: int) -> MAPFInstance:
    """Read the local Moving AI files without relying on removed helper scripts."""
    if num_agents < 0:
        raise ValueError("num_agents must be non-negative")
    lines = map_path.read_text(encoding="utf-8-sig").splitlines()
    map_start = next(index for index, line in enumerate(lines) if line.strip() == "map")
    header = dict(line.split(maxsplit=1) for line in lines[:map_start] if line.strip())
    width, height = int(header["width"]), int(header["height"])
    rows = lines[map_start + 1:]
    if len(rows) != height or any(len(row) != width for row in rows):
        raise ValueError("map dimensions do not match header")
    obstacles = set()
    for y, row in enumerate(rows):
        for x, cell in enumerate(row):
            if cell in "@OTW":
                obstacles.add((x, y))
            elif cell not in ".G":
                raise ValueError(f"unsupported map cell {cell!r} at {(x, y)}")

    scenario = scen_path.read_text(encoding="utf-8-sig").splitlines()
    records = [line.split() for line in scenario[1:] if line.strip()]
    if len(records) < num_agents:
        raise ValueError(f"scenario has fewer than {num_agents} agents")
    agents = []
    for fields in records[:num_agents]:
        # The local scenario omits the optional ninth (reference distance) field.
        if len(fields) < 8 or (int(fields[2]), int(fields[3])) != (width, height):
            raise ValueError("invalid scenario row or inconsistent map dimensions")
        if fields[1] != map_path.name:
            raise ValueError("scenario references a different map")
        agents.append(Agent((int(fields[4]), int(fields[5])),
                            (int(fields[6]), int(fields[7]))))
    return MAPFInstance(width, height, obstacles, agents)


def _validate_solution(instance: MAPFInstance, solution: Solution) -> None:
    if set(solution) != set(range(len(instance.agents))):
        raise RuntimeError("solver returned missing/extra agents")
    for agent_id, path in solution.items():
        agent = instance.agents[agent_id]
        if not path or path[0] != agent.start or path[-1] != agent.goal:
            raise RuntimeError("solver returned invalid endpoints")
        if any(not instance.traversable(cell) for cell in path):
            raise RuntimeError("solver returned an invalid map position")
        if any(abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1
               for a, b in zip(path, path[1:])):
            raise RuntimeError("solver returned a non-adjacent move")
    if find_all_conflicts(solution):
        raise RuntimeError("solver returned conflicting paths")


def run_experiment(instance: MAPFInstance) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for name, init_method, mode in VARIANTS:
        print(f"Running {name}...", file=sys.stderr, flush=True)
        config = LNSConfig(
            init_method=init_method, neighborhood_mode=mode, gamma=GAMMA,
            max_iterations=MAX_ITERATIONS, time_limit_seconds=TIME_LIMIT_SECONDS,
            seed=SEED,
        )
        row: dict[str, object] = {
            "variant": name, "init_method": init_method, "neighborhood_mode": mode,
        }
        start = time.perf_counter()
        try:
            solution, logs = run_lns(instance, config)
        except PlanningFailure as error:
            row.update(initial_soc="Failed", final_soc="", status="failed",
                       iterations=0, error=str(error))
        else:
            # The first log is measured before any repair. If no iteration ran,
            # best is the unchanged initial solution. Never run initialization twice.
            row.update(
                initial_soc=logs[0].soc_current_before if logs else solution.compute_soc(),
                final_soc=solution.compute_soc(), iterations=len(logs), error="",
                status="completed" if len(logs) == MAX_ITERATIONS else "time_limit",
            )
        row["runtime_s"] = f"{time.perf_counter() - start:.6f}"
        if row["status"] != "failed":
            _validate_solution(instance, solution)
        rows.append(row)
        print(f"Finished {name}: {row['status']}, SOC {row['initial_soc']} -> "
              f"{row['final_soc']}, {row['runtime_s']}s", file=sys.stderr, flush=True)
    return rows


def main() -> None:
    instance = load_instance(ROOT / "data/random-32-32-20.map",
                             ROOT / "data/random-32-32-20.scen", NUM_AGENTS)
    rows = run_experiment(instance)
    output = ROOT / "results_variants.csv"
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    print(output.read_text(encoding="utf-8"), end="", flush=True)


if __name__ == "__main__":
    main()
