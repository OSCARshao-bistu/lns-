"""Fixed-order Prioritized Planning using space-time A*."""

from __future__ import annotations

import heapq
from itertools import count

from .models import MAPFInstance, Path, Position, Solution


class PlanningFailure(RuntimeError):
    """Raised when Prioritized Planning cannot plan an agent in its order."""


def _position_at(path: Path, timestep: int) -> Position:
    return path[min(timestep, len(path) - 1)]


def _vertex_reserved(paths: list[Path], position: Position, timestep: int) -> bool:
    return any(_position_at(path, timestep) == position for path in paths)


def _edge_reserved(
    paths: list[Path], source: Position, target: Position, timestep: int
) -> bool:
    """Whether this move swaps across an edge used by a higher-priority agent."""
    if timestep == 0:
        return False
    return any(
        _position_at(path, timestep - 1) == target
        and _position_at(path, timestep) == source
        for path in paths
    )


def _goal_is_safe_forever(
    paths: list[Path], goal: Position, arrival_time: int
) -> bool:
    latest_reservation = max((len(path) - 1 for path in paths), default=0)
    return all(
        not _vertex_reserved(paths, goal, timestep)
        for timestep in range(arrival_time, latest_reservation + 1)
    ) and not _vertex_reserved(paths, goal, latest_reservation + 1)


def _reconstruct(
    parents: dict[tuple[Position, int], tuple[Position, int] | None],
    state: tuple[Position, int],
) -> Path:
    reversed_path: Path = []
    cursor: tuple[Position, int] | None = state
    while cursor is not None:
        reversed_path.append(cursor[0])
        cursor = parents[cursor]
    return list(reversed(reversed_path))


def _space_time_astar(
    instance: MAPFInstance, start: Position, goal: Position, reserved: list[Path]
) -> Path | None:
    if _vertex_reserved(reserved, start, 0):
        return None

    # Once all higher-priority agents have stopped, reservations are static.
    # Any useful suffix can then be chosen simple, so |V|-1 extra steps suffice.
    reservation_horizon = max((len(path) - 1 for path in reserved), default=0)
    traversable_count = instance.width * instance.height - len(instance.obstacles)
    max_timestep = reservation_horizon + max(0, traversable_count - 1)

    start_state = (start, 0)
    parents: dict[tuple[Position, int], tuple[Position, int] | None] = {
        start_state: None
    }
    best_g: dict[tuple[Position, int], int] = {start_state: 0}
    tie_breaker = count()
    frontier: list[tuple[int, int, int, Position, int]] = []
    start_h = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    heapq.heappush(frontier, (start_h, start_h, next(tie_breaker), start, 0))

    while frontier:
        _, _, _, position, timestep = heapq.heappop(frontier)
        state = (position, timestep)
        if best_g.get(state) != timestep:
            continue
        if position == goal and _goal_is_safe_forever(reserved, goal, timestep):
            return _reconstruct(parents, state)
        if timestep >= max_timestep:
            continue

        next_time = timestep + 1
        for neighbor in instance.neighbors(position):
            if _vertex_reserved(reserved, neighbor, next_time):
                continue
            if _edge_reserved(reserved, position, neighbor, next_time):
                continue
            next_state = (neighbor, next_time)
            if next_state in best_g:
                continue
            best_g[next_state] = next_time
            parents[next_state] = state
            heuristic = abs(neighbor[0] - goal[0]) + abs(neighbor[1] - goal[1])
            heapq.heappush(
                frontier,
                (next_time + heuristic, heuristic, next(tie_breaker), neighbor, next_time),
            )

    return None


def space_time_astar(
    instance: MAPFInstance, start: Position, goal: Position, reserved: list[Path]
) -> Path | None:
    """Public entry point for the same space-time A* used by PP and LNS repair."""
    return _space_time_astar(instance, start, goal, reserved)


def prioritized_planning(
    instance: MAPFInstance, agent_order: list[int] | None = None
) -> Solution:
    """Plan agents sequentially; fail rather than falling back to another solver."""
    order = list(range(len(instance.agents))) if agent_order is None else list(agent_order)
    expected = set(range(len(instance.agents)))
    if len(order) != len(expected) or set(order) != expected:
        raise ValueError("agent_order must be a permutation of all agent ids")

    solution = Solution()
    reserved: list[Path] = []
    for agent_id in order:
        agent = instance.agents[agent_id]
        path = space_time_astar(instance, agent.start, agent.goal, reserved)
        if path is None:
            raise PlanningFailure(
                f"Prioritized Planning failed for agent {agent_id} "
                f"({agent.start} -> {agent.goal})"
            )
        solution[agent_id] = path
        reserved.append(path)
    return solution
