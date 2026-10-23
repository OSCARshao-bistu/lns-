"""Standard vertex and edge-swap conflict detection."""

from __future__ import annotations

from .models import Conflict, Path, Position, Solution


def _position_at(path: Path, timestep: int) -> Position:
    if not path:
        raise ValueError("paths must contain at least one position")
    return path[min(timestep, len(path) - 1)]


def find_all_conflicts(sol: Solution) -> list[Conflict]:
    """Return conflicts ordered by time, agent pair, then vertex before edge."""
    if not sol:
        return []
    for path in sol.values():
        if not path:
            raise ValueError("paths must contain at least one position")

    conflicts: list[Conflict] = []
    agent_ids = sorted(sol)
    makespan = max(len(path) for path in sol.values()) - 1

    for timestep in range(makespan + 1):
        for index, agent_i in enumerate(agent_ids):
            path_i = sol[agent_i]
            pos_i = _position_at(path_i, timestep)
            for agent_j in agent_ids[index + 1 :]:
                path_j = sol[agent_j]
                pos_j = _position_at(path_j, timestep)
                if pos_i == pos_j:
                    conflicts.append(Conflict(agent_i, agent_j, timestep, "vertex"))

                if timestep == 0:
                    continue
                prev_i = _position_at(path_i, timestep - 1)
                prev_j = _position_at(path_j, timestep - 1)
                if prev_i == pos_j and prev_j == pos_i and prev_i != pos_i:
                    conflicts.append(Conflict(agent_i, agent_j, timestep, "edge"))

    return conflicts


def find_first_conflict(sol: Solution) -> Conflict | None:
    conflicts = find_all_conflicts(sol)
    return conflicts[0] if conflicts else None
