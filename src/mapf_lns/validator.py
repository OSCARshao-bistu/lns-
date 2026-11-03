"""Strict solution validation and shared failure reason strings.

Geometry and endpoint checks precede conflict detection. Timeouts must be
classified by the solver/runner: elapsed time cannot be inferred from paths.
"""

from __future__ import annotations

from collections.abc import Mapping

from .conflicts import find_all_conflicts
from .models import MAPFInstance, Solution


TIME_LIMIT = "time_limit"
HAS_CONFLICTS = "has_conflicts"
AGENT_UNFINISHED = "agent_unfinished"
INVALID_PATH = "invalid_path"
SOLVER_ERROR = "solver_error"
FAILURE_REASONS = frozenset({
    TIME_LIMIT, HAS_CONFLICTS, AGENT_UNFINISHED, INVALID_PATH, SOLVER_ERROR,
})


def validate_solution(instance: MAPFInstance, solution: Solution) -> tuple[bool, str]:
    """Validate a complete solution under 4-connected, stay-at-target semantics.

    Missing/empty paths or wrong final goals mean agent_unfinished. Extra/invalid
    agent IDs, wrong starts, malformed coordinates, blocked/out-of-map positions,
    diagonal moves and teleports mean invalid_path. A non-mapping solver result
    (including None) means solver_error. Vertex/edge conflicts mean has_conflicts.
    Coordinate pairs may be tuples or lists; validation never mutates its input.
    """
    if not isinstance(solution, Mapping):
        return False, SOLVER_ERROR
    expected = set(range(len(instance.agents)))
    if any(type(agent_id) is not int or agent_id not in expected for agent_id in solution):
        return False, INVALID_PATH
    if set(solution) != expected:
        return False, AGENT_UNFINISHED

    normalized = Solution()
    unfinished = False
    for agent_id, agent in enumerate(instance.agents):
        path = solution[agent_id]
        if not isinstance(path, (list, tuple)):
            return False, INVALID_PATH
        if not path:
            return False, AGENT_UNFINISHED
        checked_path = []
        for position in path:
            if (not isinstance(position, (tuple, list)) or len(position) != 2
                    or any(type(coordinate) is not int for coordinate in position)):
                return False, INVALID_PATH
            cell = (position[0], position[1])
            if not instance.traversable(cell):
                return False, INVALID_PATH
            if checked_path:
                previous = checked_path[-1]
                if abs(cell[0] - previous[0]) + abs(cell[1] - previous[1]) > 1:
                    return False, INVALID_PATH
            checked_path.append(cell)
        if checked_path[0] != agent.start:
            return False, INVALID_PATH
        if checked_path[-1] != agent.goal:
            unfinished = True
        normalized[agent_id] = checked_path

    if unfinished:
        return False, AGENT_UNFINISHED
    if find_all_conflicts(normalized):
        return False, HAS_CONFLICTS
    return True, ""
