"""Local, rule-based push-and-swap heuristic for MAPF initial solutions.

This deliberately limited PPS variant uses no graph/path search. Agents propose
moves in ID order, preferring smaller Manhattan distance. A push moves two agents
in the same tick; a swap uses three ticks around an empty grid square. Proposals
are serialized and every tick records waits for all agents that do not move.
It is incomplete: local obstacles or repeating rules may cause failure even when
a global MAPF solution exists.
"""

from __future__ import annotations

from .models import MAPFInstance, Position, Solution
from .prioritized_planning import PlanningFailure


def _local_moves(
    instance: MAPFInstance,
    positions: list[Position],
    occupied: dict[Position, int],
    agent_id: int,
    target: Position,
) -> list[dict[int, Position]] | None:
    """Return a free move, straight push, or collision-free three-tick swap."""
    source = positions[agent_id]
    blocker = occupied.get(target)
    if blocker is None:
        return [{agent_id: target}]

    dx, dy = target[0] - source[0], target[1] - source[1]
    behind = (target[0] + dx, target[1] + dy)
    if instance.traversable(behind) and behind not in occupied:
        return [{agent_id: target, blocker: behind}]

    # Direct A<->B exchange is an edge conflict. Use the two empty cells of
    # an adjacent 2x2 square instead; no temporary overlap needs later repair.
    for sx, sy in ((-dy, dx), (dy, -dx)):
        side_a = (source[0] + sx, source[1] + sy)
        side_b = (target[0] + sx, target[1] + sy)
        if all(instance.traversable(cell) and cell not in occupied
               for cell in (side_a, side_b)):
            return [
                {agent_id: side_a},
                {agent_id: side_b, blocker: source},
                {agent_id: target},
            ]
    return None


def solve_pps(instance: MAPFInstance, max_steps: int = 1000) -> Solution:
    """Generate synchronized, conflict-free paths using only local rules.

    ``max_steps`` bounds output timesteps (each swap costs three), not sweeps.
    Agents at goals may be pushed away, then resume moving toward their goals.
    Exhausting a full sweep without movement or repeating a sweep configuration
    raises ``PlanningFailure('PPS deadlock')``. No search-based fallback is used.
    """
    if max_steps < 0:
        raise ValueError("max_steps must be non-negative")

    positions = [agent.start for agent in instance.agents]
    goals = [agent.goal for agent in instance.agents]
    if len(set(positions)) != len(positions):
        raise PlanningFailure("PPS duplicate starting positions")
    if len(set(goals)) != len(goals):
        raise PlanningFailure("PPS duplicate goals cannot be occupied permanently")
    solution = Solution({agent_id: [position]
                         for agent_id, position in enumerate(positions)})
    seen: set[tuple[Position, ...]] = set()
    steps = 0

    while positions != goals:
        if steps >= max_steps:
            raise PlanningFailure(f"PPS exceeded max_steps={max_steps}")
        configuration = tuple(positions)
        if configuration in seen:
            raise PlanningFailure("PPS deadlock")
        seen.add(configuration)
        moved = False
        acted: set[int] = set()

        for agent_id, goal in enumerate(goals):
            if agent_id in acted or positions[agent_id] == goal:
                continue
            occupied = {position: other_id
                        for other_id, position in enumerate(positions)}
            source = positions[agent_id]
            candidates = [cell for cell in instance.neighbors(source) if cell != source]
            candidates.sort(key=lambda cell: abs(cell[0] - goal[0]) + abs(cell[1] - goal[1]))
            for target in candidates:
                moves = _local_moves(instance, positions, occupied, agent_id, target)
                if moves is None:
                    continue
                if steps + len(moves) > max_steps:
                    raise PlanningFailure(f"PPS exceeded max_steps={max_steps}")
                for tick in moves:
                    # A pushed agent has already acted in this sweep. Letting
                    # it immediately push back would undo the joint operation.
                    acted.update(tick)
                    for moving_id, position in tick.items():
                        positions[moving_id] = position
                    for other_id, position in enumerate(positions):
                        solution[other_id].append(position)
                    steps += 1
                moved = True
                break
            if positions == goals:
                return solution

        if not moved:
            raise PlanningFailure("PPS deadlock")

    return solution
