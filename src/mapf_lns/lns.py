"""MAPF-LNS with mixed, fixed, or adaptive neighborhood selection.

Line mapping: initial PP (line 1), copy P_best (line 2), seeded RNG and baseline
heuristics (lines 3-4), bounded loop/timeout (lines 5-6), uniform heuristic and
destroy (lines 7-8), PP repair/failure rejection (lines 9-10), best/current
greedy updates (lines 11-12), logging (line 13), and return (line 14).
"""

from __future__ import annotations

from dataclasses import dataclass
import random
import time

from .destroy import AgentBasedDestroy, NeighborhoodDestroy, RandomDestroy
from . import eecbs, pps
from .models import MAPFInstance, Solution
from .prioritized_planning import PlanningFailure, prioritized_planning
from .repair import repair_prioritized


@dataclass(frozen=True, slots=True)
class LNSIterationLog:
    iteration: int
    heuristic: str
    soc_current_before: int
    soc_candidate: int | None
    accepted: bool
    soc_best_after: int
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class LNSConfig:
    max_iterations: int = 1000
    gamma: float = 0.1
    seed: int = 0
    time_limit_seconds: float | None = None
    neighborhood_mode: str = "random"
    init_method: str = "pp"

    def __post_init__(self) -> None:
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in the interval (0, 1]")
        if self.time_limit_seconds is not None and self.time_limit_seconds < 0:
            raise ValueError("time_limit_seconds must be non-negative or None")
        if self.neighborhood_mode not in {"random", "agent_based", "map_based", "alns"}:
            raise ValueError("unsupported neighborhood_mode: " + self.neighborhood_mode)
        if self.init_method not in {"pp", "pps", "eecbs"}:
            raise ValueError("unsupported init_method: " + self.init_method)


class _AdaptiveSelector:
    """Roulette selection with rewards averaged over 50-iteration segments.

    Empty neighborhoods receive zero reward, like unsuccessful repairs. Weights
    persist across segments; scores and selection counts reset after each update.
    """

    def __init__(self, operator_count: int) -> None:
        self.weights = [1.0] * operator_count
        self.scores = [0.0] * operator_count
        self.counts = [0] * operator_count
        self.iterations = 0

    def select(self, rng: random.Random) -> int:
        threshold = rng.random() * sum(self.weights)
        cumulative = 0.0
        for index, weight in enumerate(self.weights):
            cumulative += weight
            if threshold < cumulative:
                return index
        return len(self.weights) - 1  # Floating-point rounding at the upper bound.

    def record(self, index: int, score: int) -> None:
        self.scores[index] += score
        self.counts[index] += 1
        self.iterations += 1
        if self.iterations % 50 == 0:
            for operator, count in enumerate(self.counts):
                if count:
                    self.weights[operator] = (
                        0.9 * self.weights[operator]
                        + 0.1 * self.scores[operator] / count
                    )
            self.scores = [0.0] * len(self.weights)
            self.counts = [0] * len(self.weights)


def run_lns(
    instance: MAPFInstance, config: LNSConfig
) -> tuple[Solution, list[LNSIterationLog]]:
    """Run MAPF-LNS; the loop time limit excludes initial solution generation."""
    if config.init_method == "pps":
        current = pps.solve_pps(instance)
    elif config.init_method == "eecbs":
        current = eecbs.solve_eecbs(instance)
    else:
        current = prioritized_planning(instance)  # Algorithm 1, line 1
    best = Solution(current.copy())  # Algorithm 1, line 2
    rng = random.Random(config.seed)  # Algorithm 1, line 3
    heuristics = [
        RandomDestroy(),
        AgentBasedDestroy(),
        NeighborhoodDestroy(),
    ]  # Algorithm 1, line 4
    adaptive = _AdaptiveSelector(len(heuristics)) if config.neighborhood_mode == "alns" else None
    fixed_index = {"agent_based": 1, "map_based": 2}.get(config.neighborhood_mode)
    logs: list[LNSIterationLog] = []
    loop_start = time.perf_counter()

    for iteration in range(config.max_iterations):  # Algorithm 1, line 5
        if (
            config.time_limit_seconds is not None
            and time.perf_counter() - loop_start >= config.time_limit_seconds
        ):  # Algorithm 1, line 6
            break

        if adaptive is not None:
            selected_index = adaptive.select(rng)
            heuristic = heuristics[selected_index]
        elif fixed_index is not None:
            heuristic = heuristics[fixed_index]
        else:
            heuristic = rng.choice(heuristics)  # Preserve the default RNG sequence.
        to_replan = heuristic.destroy(
            instance, current, rng, config.gamma
        )  # Algorithm 1, line 8
        soc_before = current.compute_soc()

        if not to_replan:
            if adaptive is not None:
                adaptive.record(selected_index, 0)
            logs.append(
                LNSIterationLog(
                    iteration=iteration,
                    heuristic=heuristic.name,
                    soc_current_before=soc_before,
                    soc_candidate=None,
                    accepted=False,
                    soc_best_after=best.compute_soc(),
                    failure_reason="neighborhood_unavailable",
                )
            )
            continue

        try:
            candidate = repair_prioritized(
                instance, current, to_replan, rng
            )  # Algorithm 1, line 9
        except PlanningFailure:  # Algorithm 1, line 10
            if adaptive is not None:
                adaptive.record(selected_index, 0)
            logs.append(
                LNSIterationLog(
                    iteration=iteration,
                    heuristic=heuristic.name,
                    soc_current_before=soc_before,
                    soc_candidate=None,
                    accepted=False,
                    soc_best_after=best.compute_soc(),
                    failure_reason="repair_failed",
                )
            )
            continue

        candidate_soc = candidate.compute_soc()
        improved_best = candidate_soc < best.compute_soc()
        if improved_best:  # Algorithm 1, line 11
            best = Solution(candidate.copy())

        accepted = candidate_soc <= soc_before  # Algorithm 1, line 12
        if accepted:
            current = candidate
        if adaptive is not None:
            adaptive.record(selected_index, 33 if improved_best else 9 if accepted else 1)

        logs.append(  # Algorithm 1, line 13
            LNSIterationLog(
                iteration=iteration,
                heuristic=heuristic.name,
                soc_current_before=soc_before,
                soc_candidate=candidate_soc,
                accepted=accepted,
                soc_best_after=best.compute_soc(),
            )
        )

    return best, logs  # Algorithm 1, line 14
