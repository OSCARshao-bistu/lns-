"""Local Repair A* (LRA*) baseline solver."""

from __future__ import annotations

import heapq
import random
from itertools import count

from .conflicts import find_first_conflict
from .models import MAPFInstance, Path, Position, Solution
from .prioritized_planning import PlanningFailure, space_time_astar


def _space_only_astar(
    instance: MAPFInstance, start: Position, goal: Position
) -> Path:
    """Find a shortest static-grid path, ignoring all other agents and wait."""
    if not instance.traversable(start) or not instance.traversable(goal):
        raise PlanningFailure(f"no spatial path from {start} to {goal}")

    parents: dict[Position, Position | None] = {start: None}
    best_g: dict[Position, int] = {start: 0}
    tie_breaker = count()
    frontier: list[tuple[int, int, int, Position]] = []
    start_h = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    heapq.heappush(frontier, (start_h, start_h, next(tie_breaker), start))

    while frontier:
        _, _, _, position = heapq.heappop(frontier)
        current_g = best_g[position]
        if position == goal:
            reversed_path: Path = []
            cursor: Position | None = position
            while cursor is not None:
                reversed_path.append(cursor)
                cursor = parents[cursor]
            return list(reversed(reversed_path))

        x, y = position
        # Pure spatial A*: four-connected moves only, deliberately no wait action.
        for neighbor in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
            if not instance.traversable(neighbor):
                continue
            tentative_g = current_g + 1
            if tentative_g >= best_g.get(neighbor, tentative_g + 1):
                continue
            best_g[neighbor] = tentative_g
            parents[neighbor] = position
            heuristic = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
            heapq.heappush(
                frontier,
                (
                    tentative_g + heuristic,
                    heuristic,
                    next(tie_breaker),
                    neighbor,
                ),
            )

    raise PlanningFailure(f"no spatial path from {start} to {goal}")


def solve_lra_star(
    instance: MAPFInstance, max_iterations: int = 500
) -> Solution:
    """Solve MAPF by independent planning followed by one-agent local repairs."""
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")

    solution = Solution(
        {
            agent_id: _space_only_astar(instance, agent.start, agent.goal)
            for agent_id, agent in enumerate(instance.agents)
        }
    )

    for _ in range(max_iterations):
        conflict = find_first_conflict(solution)
        if conflict is None:
            return solution

        repair_order = [conflict.agent_i, conflict.agent_j]
        if random.choice((False, True)):
            repair_order.reverse()

        repaired = False
        for agent_id in repair_order:
            old_path = solution.pop(agent_id)
            agent = instance.agents[agent_id]
            reserved = list(solution.values())
            new_path = space_time_astar(
                instance, agent.start, agent.goal, reserved
            )
            if new_path is not None:
                solution[agent_id] = new_path
                repaired = True
                break
            solution[agent_id] = old_path

        if not repaired:
            raise PlanningFailure("LRA* stuck in deadlock")

    if find_first_conflict(solution) is None:
        return solution
    raise PlanningFailure(
        f"LRA* still has conflicts after {max_iterations} repair iterations"
    )
