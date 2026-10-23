"""Data structures shared by the MAPF algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

Position: TypeAlias = tuple[int, int]
Path: TypeAlias = list[Position]


@dataclass(frozen=True, slots=True)
class Agent:
    start: Position
    goal: Position


@dataclass(frozen=True, slots=True)
class MAPFInstance:
    width: int
    height: int
    obstacles: set[Position]
    agents: list[Agent]

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")

        invalid_obstacles = [p for p in self.obstacles if not self.in_bounds(p)]
        if invalid_obstacles:
            raise ValueError(f"obstacles outside the grid: {invalid_obstacles}")

        for agent_id, agent in enumerate(self.agents):
            for label, position in (("start", agent.start), ("goal", agent.goal)):
                if not self.in_bounds(position):
                    raise ValueError(
                        f"agent {agent_id} {label} is outside the grid: {position}"
                    )
                if position in self.obstacles:
                    raise ValueError(
                        f"agent {agent_id} {label} lies on an obstacle: {position}"
                    )

    def in_bounds(self, position: Position) -> bool:
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def traversable(self, position: Position) -> bool:
        return self.in_bounds(position) and position not in self.obstacles

    def neighbors(self, position: Position) -> list[Position]:
        """Return 4-connected moves plus wait, in a deterministic order."""
        x, y = position
        candidates = [(x, y), (x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)]
        return [candidate for candidate in candidates if self.traversable(candidate)]


class Solution(dict[int, Path]):
    """Mapping from agent id to path; an agent stays at its final location."""

    def compute_soc(self) -> int:
        if any(not path for path in self.values()):
            raise ValueError("paths must contain at least one position")
        return sum(len(path) - 1 for path in self.values())


@dataclass(frozen=True, slots=True)
class Conflict:
    agent_i: int
    agent_j: int
    timestep: int
    type: Literal["vertex", "edge"]
