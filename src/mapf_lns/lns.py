"""Li et al. 2021 baseline MAPF-LNS, corresponding to Algorithm 1.

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


@dataclass(frozen=True, slots=True)
class LNSConfig:
    max_iterations: int = 1000
    gamma: float = 0.1
    seed: int = 0
    time_limit_seconds: float | None = None

    def __post_init__(self) -> None:
        if self.max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in the interval (0, 1]")
        if self.time_limit_seconds is not None and self.time_limit_seconds < 0:
            raise ValueError("time_limit_seconds must be non-negative or None")


def run_lns(
    instance: MAPFInstance, config: LNSConfig
) -> tuple[Solution, list[LNSIterationLog]]:
    """Run non-adaptive MAPF-LNS with PP repair and greedy acceptance."""
    current = prioritized_planning(instance)  # Algorithm 1, line 1
    best = Solution(current.copy())  # Algorithm 1, line 2
    rng = random.Random(config.seed)  # Algorithm 1, line 3
    heuristics = [
        RandomDestroy(),
        AgentBasedDestroy(),
        NeighborhoodDestroy(),
    ]  # Algorithm 1, line 4
    logs: list[LNSIterationLog] = []
    loop_start = time.perf_counter()

    for iteration in range(config.max_iterations):  # Algorithm 1, line 5
        if (
            config.time_limit_seconds is not None
            and time.perf_counter() - loop_start >= config.time_limit_seconds
        ):  # Algorithm 1, line 6
            break

        heuristic = rng.choice(heuristics)  # Algorithm 1, line 7
        to_replan = heuristic.destroy(
            instance, current, rng, config.gamma
        )  # Algorithm 1, line 8
        soc_before = current.compute_soc()

        try:
            candidate = repair_prioritized(
                instance, current, to_replan, rng
            )  # Algorithm 1, line 9
        except PlanningFailure:  # Algorithm 1, line 10
            logs.append(
                LNSIterationLog(
                    iteration=iteration,
                    heuristic=heuristic.name,
                    soc_current_before=soc_before,
                    soc_candidate=None,
                    accepted=False,
                    soc_best_after=best.compute_soc(),
                )
            )
            continue

        candidate_soc = candidate.compute_soc()
        if candidate_soc < best.compute_soc():  # Algorithm 1, line 11
            best = Solution(candidate.copy())

        accepted = candidate_soc <= soc_before  # Algorithm 1, line 12
        if accepted:
            current = candidate

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
