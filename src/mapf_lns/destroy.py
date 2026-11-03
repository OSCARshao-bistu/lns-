"""Destroy heuristics for the non-adaptive MAPF-LNS baseline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
import math
import random

from .models import MAPFInstance, Path, Position, Solution


def _neighborhood_size(agent_count: int, gamma: float) -> int:
    if agent_count == 0:
        return 0
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in the interval (0, 1]")
    return max(1, math.ceil(gamma * agent_count))


def _position_at(path: Path, timestep: int) -> Position:
    return path[min(timestep, len(path) - 1)]


def _blocking_agents(
    solution: Solution, agent_id: int, source: Position,
    target: Position, arrival_time: int,
) -> set[int]:
    """Find blockers of a hypothetical move, without changing any real path.

    Keep this project's permanent goal occupancy, including after stored paths
    end. The original C++ PathTable collector leaves this case as a TODO.
    """
    blockers = set()
    for other_id, path in solution.items():
        if other_id == agent_id:
            continue
        current = _position_at(path, arrival_time)
        previous = _position_at(path, arrival_time - 1)
        if current == target or (
            source != target and previous == target and current == source
        ):
            blockers.add(other_id)
    return blockers


def _bfs_hops(instance: MAPFInstance, source: Position) -> dict[Position, int]:
    """Return 4-connected unweighted hop distances to reachable map cells."""
    distances = {source: 0}
    frontier = deque([source])
    while frontier:
        x, y = frontier.popleft()
        next_distance = distances[(x, y)] + 1
        for neighbor in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
            if neighbor in distances or not instance.traversable(neighbor):
                continue
            distances[neighbor] = next_distance
            frontier.append(neighbor)
    return distances


class DestroyHeuristic(ABC):
    name: str

    @abstractmethod
    def destroy(
        self,
        instance: MAPFInstance,
        solution: Solution,
        rng: random.Random,
        gamma: float,
    ) -> set[int]:
        """Return the agent ids to replan in this iteration."""


class RandomDestroy(DestroyHeuristic):
    name = "random"

    def destroy(
        self,
        instance: MAPFInstance,
        solution: Solution,
        rng: random.Random,
        gamma: float,
    ) -> set[int]:
        # Baseline only: no adaptive weights, learning, bandits, or delay signals.
        size = _neighborhood_size(len(instance.agents), gamma)
        return set(rng.sample(range(len(instance.agents)), size))


class AgentBasedDestroy(DestroyHeuristic):
    """Li et al.'s agent-based restricted random walk, with a persistent tabu list.

    Port of generateNeighborByRandomWalk/findMostDelayedAgent/randomWalk:
    https://github.com/Jiaoyang-Li/MAPF-LNS/blob/master/src/LNS.cpp
    An empty set signals no usable neighborhood; no random padding is applied.
    """

    name = "agent_based"

    def __init__(self) -> None:
        self._tabu: set[int] = set()
        self._instance: MAPFInstance | None = None
        self._distance_maps: dict[Position, dict[Position, int]] = {}

    def _distances(self, instance: MAPFInstance, goal: Position) -> dict[Position, int]:
        if goal not in self._distance_maps:
            self._distance_maps[goal] = _bfs_hops(instance, goal)
        return self._distance_maps[goal]

    def _select_seed(self, instance: MAPFInstance, solution: Solution) -> int | None:
        seed = None
        max_delay = -1
        for agent_id, agent in enumerate(instance.agents):
            if agent_id in self._tabu:
                continue
            shortest = self._distances(instance, agent.goal).get(agent.start, math.inf)
            delay = len(solution[agent_id]) - 1 - shortest
            if delay > max_delay:
                seed, max_delay = agent_id, delay
        if seed is None or max_delay <= 0:
            self._tabu.clear()
            return None
        self._tabu.add(seed)
        if len(self._tabu) == len(instance.agents):
            self._tabu.clear()
        return seed

    def _walk(
        self, instance: MAPFInstance, solution: Solution, agent_id: int,
        start_time: int, selected: set[int], size: int, rng: random.Random,
    ) -> None:
        path = solution[agent_id]
        position = path[start_time]
        upper_bound = len(path) - 1
        distances = self._distances(instance, instance.agents[agent_id].goal)
        for timestep in range(start_time, upper_bound):
            candidates = instance.neighbors(position)
            while candidates:
                target = rng.choice(candidates)
                if timestep + 1 + distances.get(target, math.inf) < upper_bound:
                    blockers = _blocking_agents(
                        solution, agent_id, position, target, timestep + 1
                    )
                    # Keep the first k agents; sorted IDs make ties reproducible.
                    for blocker in sorted(blockers):
                        if len(selected) >= size:
                            break
                        selected.add(blocker)
                    position = target
                    break
                candidates.remove(target)
            if not candidates or len(selected) >= size:
                break

    def destroy(
        self,
        instance: MAPFInstance,
        solution: Solution,
        rng: random.Random,
        gamma: float,
    ) -> set[int]:
        if self._instance is not instance:
            self._instance = instance
            self._distance_maps.clear()
            self._tabu.clear()
        agent_count = len(instance.agents)
        size = _neighborhood_size(agent_count, gamma)
        if size == 0:
            return set()
        if size >= agent_count:
            return set(range(agent_count))
        seed = self._select_seed(instance, solution)
        if seed is None:
            return set()
        selected = {seed}
        self._walk(instance, solution, seed, 0, selected, size, rng)
        agent_id = seed
        for _ in range(10):
            if len(selected) >= size:
                break
            start_time = rng.randrange(len(solution[agent_id]))
            self._walk(instance, solution, agent_id, start_time, selected, size, rng)
            agent_id = rng.choice(sorted(selected))
        # Match the reference failure behavior, including a size-one target.
        return selected if len(selected) >= 2 else set()


class NeighborhoodDestroy(DestroyHeuristic):
    name = "neighborhood"

    def destroy(
        self,
        instance: MAPFInstance,
        solution: Solution,
        rng: random.Random,
        gamma: float,
    ) -> set[int]:
        # Baseline only: no adaptive weights, learning, bandits, or delay signals.
        agent_count = len(instance.agents)
        size = _neighborhood_size(agent_count, gamma)
        if size == 0:
            return set()

        seed = rng.choice(range(agent_count))
        hop_distances = _bfs_hops(instance, instance.agents[seed].start)

        def distance_to_path(agent_id: int) -> float:
            return min(
                (hop_distances[position] for position in solution[agent_id] if position in hop_distances),
                default=math.inf,
            )

        ordered = sorted(
            (agent_id for agent_id in range(agent_count) if agent_id != seed),
            key=lambda agent_id: (distance_to_path(agent_id), agent_id),
        )
        return {seed, *ordered[: size - 1]}
