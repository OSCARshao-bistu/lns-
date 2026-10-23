"""Destroy heuristics for the non-adaptive MAPF-LNS baseline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
import math
import random

from .conflicts import find_all_conflicts
from .models import MAPFInstance, Position, Solution


def _neighborhood_size(agent_count: int, gamma: float) -> int:
    if agent_count == 0:
        return 0
    if not 0 < gamma <= 1:
        raise ValueError("gamma must be in the interval (0, 1]")
    return max(1, math.ceil(gamma * agent_count))


def _conflict_graph(solution: Solution, agent_count: int) -> dict[int, set[int]]:
    graph = {agent_id: set() for agent_id in range(agent_count)}
    for conflict in find_all_conflicts(solution):
        graph[conflict.agent_i].add(conflict.agent_j)
        graph[conflict.agent_j].add(conflict.agent_i)
    return graph


def _random_walk_selection(
    graph: dict[int, set[int]], seed: int, target_size: int, rng: random.Random
) -> set[int]:
    selected = {seed}
    frontier = set(graph[seed])

    while len(selected) < target_size and frontier:
        candidate = rng.choice(sorted(frontier))
        frontier.remove(candidate)
        if candidate in selected:
            continue
        selected.add(candidate)
        frontier.update(graph[candidate] - selected)

    if len(selected) < target_size:
        remaining = sorted(set(graph) - selected)
        selected.update(rng.sample(remaining, target_size - len(selected)))
    return selected


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
    name = "agent_based"

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
        return _random_walk_selection(
            _conflict_graph(solution, agent_count), seed, size, rng
        )


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
