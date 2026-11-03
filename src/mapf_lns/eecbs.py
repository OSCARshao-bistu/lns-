"""CBS with exact constrained A* and the requested high-level focal search.

The public name follows the experiment interface. This implements the specified
ECBS-style variant, not the additional explicit-estimation mechanisms of full
EECBS. ``weight`` is additive slack: FOCAL admits cost <= (1 + weight) * min_cost.
Costs count moves/waits (len(path) - 1), matching Solution.compute_soc().
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from itertools import count
import math
import time

from .conflicts import find_all_conflicts
from .models import Conflict, MAPFInstance, Path, Position, Solution
from .prioritized_planning import PlanningFailure


@dataclass(frozen=True, slots=True)
class _VertexConstraint:
    position: Position
    timestep: int


@dataclass(frozen=True, slots=True)
class _EdgeConstraint:
    source: Position
    target: Position
    timestep: int  # Departure time; arrival is timestep + 1.


_Constraint = _VertexConstraint | _EdgeConstraint
_State = tuple[int, int, int]


def _check_deadline(deadline: float) -> None:
    if time.perf_counter() >= deadline:
        raise PlanningFailure("EECBS time limit exceeded")


def _constrained_astar(
    instance: MAPFInstance,
    agent_id: int,
    constraints: frozenset[_Constraint] = frozenset(),
    deadline: float = math.inf,
) -> tuple[Path, int] | None:
    """Return a shortest legal path and move cost, or None if none exists.

    After the last constraint, a simple spatial suffix of at most |V|-1 moves
    suffices. This finite horizon avoids searching forever for disconnected goals.
    Reaching a goal is terminal only when all future goal/wait constraints permit
    staying there forever; merely visiting the goal earlier remains legal.
    """
    _check_deadline(deadline)
    agent = instance.agents[agent_id]
    vertices = {(c.position, c.timestep) for c in constraints
                if isinstance(c, _VertexConstraint)}
    edges = {(c.source, c.target, c.timestep) for c in constraints
             if isinstance(c, _EdgeConstraint)}
    if (agent.start, 0) in vertices:
        return None
    latest = max(
        [0] + [t for _, t in vertices] + [t + 1 for _, _, t in edges]
    )
    goal_safe_from = max(
        [0] + [t + 1 for position, t in vertices if position == agent.goal]
        + [t + 1 for source, target, t in edges if source == target == agent.goal]
    )
    free_cells = instance.width * instance.height - len(instance.obstacles)
    horizon = latest + max(0, free_cells - 1)
    start: _State = (*agent.start, 0)
    parents: dict[_State, _State | None] = {start: None}
    serial = count()
    start_h = abs(agent.start[0] - agent.goal[0]) + abs(agent.start[1] - agent.goal[1])
    frontier = [(start_h, start_h, next(serial), start)]

    while frontier:
        _check_deadline(deadline)
        _, _, _, state = heapq.heappop(frontier)
        x, y, timestep = state
        position = (x, y)
        if position == agent.goal and timestep >= goal_safe_from:
            path: Path = []
            cursor: _State | None = state
            while cursor is not None:
                path.append((cursor[0], cursor[1]))
                cursor = parents[cursor]
            path.reverse()
            return path, len(path) - 1
        if timestep >= horizon:
            continue
        next_time = timestep + 1
        for target in instance.neighbors(position):
            if (target, next_time) in vertices or (position, target, timestep) in edges:
                continue
            next_state: _State = (*target, next_time)
            # At a fixed (x,y,t), g is always t; rediscovery cannot improve it.
            if next_state in parents:
                continue
            parents[next_state] = state
            heuristic = abs(target[0] - agent.goal[0]) + abs(target[1] - agent.goal[1])
            heapq.heappush(frontier, (next_time + heuristic, heuristic,
                                      next(serial), next_state))
    return None


@dataclass(frozen=True, slots=True)
class _CTNode:
    constraints: tuple[frozenset[_Constraint], ...]
    solution: Solution
    cost: int
    conflicts: tuple[Conflict, ...]


class _FocalQueue:
    """Cost-ordered OPEN and conflict-ordered FOCAL with stable tie breaking."""

    def __init__(self, weight: float) -> None:
        self.factor = 1 + weight
        self._serial = count()
        self._open: list[tuple[int, int]] = []
        self._active: dict[int, _CTNode] = {}

    def push(self, node: _CTNode) -> None:
        serial = next(self._serial)
        self._active[serial] = node
        heapq.heappush(self._open, (node.cost, serial))

    def __bool__(self) -> bool:
        return bool(self._active)

    def pop(self) -> _CTNode:
        while self._open and self._open[0][1] not in self._active:
            heapq.heappop(self._open)
        if not self._open:
            raise IndexError("empty FOCAL queue")
        min_cost = self._open[0][0]
        # Rebuild membership from the CURRENT OPEN minimum every time. This also
        # removes stale nodes and admits new nodes as the lower bound increases.
        focal = [(len(node.conflicts), node.cost, serial)
                 for serial, node in self._active.items()
                 if node.cost <= self.factor * min_cost]
        heapq.heapify(focal)
        _, _, serial = heapq.heappop(focal)
        return self._active.pop(serial)


def _position_at(path: Path, timestep: int) -> Position:
    return path[min(timestep, len(path) - 1)]


def solve_eecbs(
    instance: MAPFInstance,
    time_limit_seconds: float = 5.0,
    weight: float = 1.5,
) -> Solution:
    """Solve using conflict-tree branching and high-level focal selection.

    The shared deadline covers root planning, CT expansion, and low-level A*.
    Failure means proven infeasibility (e.g. exhausted OPEN) or time exhaustion;
    a time-limit failure does not imply the MAPF instance is unsolvable.
    """
    if not math.isfinite(time_limit_seconds) or time_limit_seconds < 0:
        raise ValueError("time_limit_seconds must be finite and non-negative")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and non-negative")
    deadline = time.perf_counter() + time_limit_seconds
    _check_deadline(deadline)
    if len({agent.start for agent in instance.agents}) != len(instance.agents):
        raise PlanningFailure("EECBS no solution: duplicate starts")
    if len({agent.goal for agent in instance.agents}) != len(instance.agents):
        raise PlanningFailure("EECBS no solution: duplicate permanent goals")

    constraints: tuple[frozenset[_Constraint], ...] = tuple(
        frozenset() for _ in instance.agents
    )
    solution = Solution()
    for agent_id in range(len(instance.agents)):
        result = _constrained_astar(instance, agent_id, constraints[agent_id], deadline)
        if result is None:
            raise PlanningFailure(f"EECBS no solution: agent {agent_id} cannot reach goal")
        solution[agent_id] = result[0]
    root = _CTNode(constraints, solution, solution.compute_soc(),
                   tuple(find_all_conflicts(solution)))
    frontier = _FocalQueue(weight)
    frontier.push(root)
    seen = {constraints}

    while frontier:
        _check_deadline(deadline)
        node = frontier.pop()
        _check_deadline(deadline)
        if not node.conflicts:
            return node.solution
        conflict = node.conflicts[0]  # Existing detector orders by time then pair.
        for agent_id in (conflict.agent_i, conflict.agent_j):
            _check_deadline(deadline)
            path = node.solution[agent_id]
            target = _position_at(path, conflict.timestep)
            added: _Constraint
            if conflict.type == "vertex":
                added = _VertexConstraint(target, conflict.timestep)
            else:
                source = _position_at(path, conflict.timestep - 1)
                added = _EdgeConstraint(source, target, conflict.timestep - 1)
            child_constraints = list(node.constraints)
            child_constraints[agent_id] = child_constraints[agent_id] | {added}
            key = tuple(child_constraints)
            if key in seen:
                continue
            seen.add(key)
            result = _constrained_astar(instance, agent_id, key[agent_id], deadline)
            if result is None:
                continue
            child_solution = Solution(node.solution.copy())
            child_solution[agent_id] = result[0]
            child = _CTNode(key, child_solution, child_solution.compute_soc(),
                            tuple(find_all_conflicts(child_solution)))
            _check_deadline(deadline)
            frontier.push(child)

    raise PlanningFailure("EECBS no solution: constraint tree exhausted")
