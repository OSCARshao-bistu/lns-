"""Prioritized Planning repair for a destroyed MAPF-LNS neighborhood."""

from __future__ import annotations

import random

from .models import MAPFInstance, Solution
from .prioritized_planning import PlanningFailure, space_time_astar


def repair_prioritized(
    instance: MAPFInstance,
    solution: Solution,
    to_replan: set[int],
    rng: random.Random,
) -> Solution:
    """Keep all other paths fixed and replan selected agents in random order."""
    valid_ids = set(range(len(instance.agents)))
    if not to_replan <= valid_ids:
        raise ValueError("to_replan contains an invalid agent id")

    repaired = Solution(solution.copy())
    reserved = [
        solution[agent_id]
        for agent_id in range(len(instance.agents))
        if agent_id not in to_replan
    ]

    replan_order = sorted(to_replan)
    rng.shuffle(replan_order)
    for agent_id in replan_order:
        agent = instance.agents[agent_id]
        path = space_time_astar(instance, agent.start, agent.goal, reserved)
        if path is None:
            raise PlanningFailure(
                f"Prioritized repair failed for agent {agent_id} "
                f"({agent.start} -> {agent.goal})"
            )
        repaired[agent_id] = path
        reserved.append(path)

    return repaired
