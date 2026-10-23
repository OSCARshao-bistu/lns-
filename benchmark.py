"""Reproducible benchmark instance generators for baseline MAPF-LNS."""

from __future__ import annotations

from collections import deque
import random

from mapf_lns import Agent, MAPFInstance, Position


def _sample_agents(
    cells: list[Position], num_agents: int, rng: random.Random
) -> list[Agent]:
    if 2 * num_agents > len(cells):
        raise ValueError("not enough distinct free cells for all starts and goals")
    endpoints = rng.sample(cells, 2 * num_agents)
    starts = endpoints[:num_agents]
    goals = endpoints[num_agents:]
    return [Agent(start, goal) for start, goal in zip(starts, goals)]


def _connected_free_cells(
    size: int, obstacles: set[Position]
) -> list[Position] | None:
    free = [
        (x, y)
        for y in range(size)
        for x in range(size)
        if (x, y) not in obstacles
    ]
    if not free:
        return None

    reached = {free[0]}
    frontier = deque([free[0]])
    while frontier:
        x, y = frontier.popleft()
        for neighbor in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
            nx, ny = neighbor
            if (
                0 <= nx < size
                and 0 <= ny < size
                and neighbor not in obstacles
                and neighbor not in reached
            ):
                reached.add(neighbor)
                frontier.append(neighbor)
    return free if len(reached) == len(free) else None


def make_empty(size: int, num_agents: int, seed: int) -> MAPFInstance:
    """Create an empty square grid with distinct random starts and goals."""
    if size <= 0 or num_agents < 0:
        raise ValueError("size must be positive and num_agents non-negative")
    rng = random.Random(seed)
    cells = [(x, y) for y in range(size) for x in range(size)]
    return MAPFInstance(size, size, set(), _sample_agents(cells, num_agents, rng))


def make_random_obstacles(
    size: int, num_agents: int, obstacle_ratio: float, seed: int
) -> MAPFInstance:
    """Create a connected random-obstacle grid with distinct endpoints."""
    if size <= 0 or num_agents < 0:
        raise ValueError("size must be positive and num_agents non-negative")
    if not 0 <= obstacle_ratio < 1:
        raise ValueError("obstacle_ratio must be in the interval [0, 1)")

    cell_count = size * size
    obstacle_count = round(obstacle_ratio * cell_count)
    if cell_count - obstacle_count < 2 * num_agents:
        raise ValueError("not enough free cells for all distinct starts and goals")

    rng = random.Random(seed)
    cells = [(x, y) for y in range(size) for x in range(size)]
    # Resampling concerns map connectivity only; PP is never retried here.
    for _ in range(10_000):
        obstacles = set(rng.sample(cells, obstacle_count))
        free = _connected_free_cells(size, obstacles)
        if free is not None:
            return MAPFInstance(
                size, size, obstacles, _sample_agents(free, num_agents, rng)
            )
    raise RuntimeError("could not sample a connected random-obstacle map")


def make_corridor_swap(num_agents: int, seed: int) -> MAPFInstance:
    """Create opposing traffic in a one-cell corridor with sparse passing bays."""
    if num_agents <= 0:
        raise ValueError("num_agents must be positive")

    rng = random.Random(seed)
    width = 2 * num_agents + 3
    height = 3
    corridor = {(x, 1) for x in range(width)}
    bays = {(x, 0) for x in range(2, width - 2, 3)}
    free = corridor | bays
    obstacles = {
        (x, y) for y in range(height) for x in range(width) if (x, y) not in free
    }

    left = [(x, 1) for x in range(1, num_agents + 1)]
    right = [(width - 2 - x, 1) for x in range(num_agents)]
    split = (num_agents + 1) // 2
    starts = left[:split] + right[: num_agents - split]
    goals = right[:split] + left[: num_agents - split]

    # Seed controls which physical agents receive the opposing endpoint pairs.
    pairs = list(zip(starts, goals))
    rng.shuffle(pairs)
    agents = [Agent(start, goal) for start, goal in pairs]
    return MAPFInstance(width, height, obstacles, agents)


def make_dense_empty(size: int, num_agents: int, seed: int) -> MAPFInstance:
    """Create an empty grid with at least 30 percent agent density."""
    if size <= 0 or num_agents < 0:
        raise ValueError("size must be positive and num_agents non-negative")
    if num_agents < 0.3 * size * size:
        raise ValueError("dense scenarios require at least 30 percent agent density")
    return make_empty(size, num_agents, seed)


def make_bottleneck(size: int, num_agents: int, seed: int) -> MAPFInstance:
    """Create opposing regions separated by a wall with two one-cell gaps."""
    if size < 4 or num_agents < 0:
        raise ValueError("size must be at least 4 and num_agents non-negative")
    rng = random.Random(seed)
    wall_y = size // 2
    gaps = {size // 4, 3 * size // 4}
    obstacles = {(x, wall_y) for x in range(size) if x not in gaps}
    upper = [(x, y) for y in range(wall_y) for x in range(size)]
    lower = [(x, y) for y in range(wall_y + 1, size) for x in range(size)]
    if num_agents > len(upper) or num_agents > len(lower):
        raise ValueError("not enough cells on each side for all agents")
    starts = rng.sample(upper, num_agents)
    goals = rng.sample(lower, num_agents)
    return MAPFInstance(
        size,
        size,
        obstacles,
        [Agent(start, goal) for start, goal in zip(starts, goals)],
    )


def make_hub(size: int, num_agents: int, seed: int) -> MAPFInstance:
    """Create four isolated quadrants connected to a central hub by one gate each."""
    if size < 9 or num_agents < 0:
        raise ValueError("size must be at least 9 and num_agents non-negative")
    rng = random.Random(seed)
    hub_size = size // 3
    low = (size - hub_size) // 2
    high = low + hub_size - 1

    quadrants = [
        [(x, y) for y in range(low) for x in range(low)],
        [(x, y) for y in range(low) for x in range(high + 1, size)],
        [(x, y) for y in range(high + 1, size) for x in range(low)],
        [
            (x, y)
            for y in range(high + 1, size)
            for x in range(high + 1, size)
        ],
    ]
    hub = {
        (x, y)
        for y in range(low, high + 1)
        for x in range(low, high + 1)
    }
    gates = {(low - 1, low), (high + 1, low), (low - 1, high), (high + 1, high)}
    free = hub | gates | {cell for quadrant in quadrants for cell in quadrant}
    obstacles = {
        (x, y) for y in range(size) for x in range(size) if (x, y) not in free
    }

    pools = [list(cells) for cells in quadrants]
    for pool in pools:
        rng.shuffle(pool)
    start_quadrants = [agent_id % 4 for agent_id in range(num_agents)]
    rng.shuffle(start_quadrants)
    starts: list[Position] = []
    for quadrant_id in start_quadrants:
        if not pools[quadrant_id]:
            raise ValueError("not enough quadrant cells for all starts")
        starts.append(pools[quadrant_id].pop())

    goals: list[Position] = []
    for start_quadrant in start_quadrants:
        choices = [
            quadrant_id
            for quadrant_id, pool in enumerate(pools)
            if quadrant_id != start_quadrant and pool
        ]
        if not choices:
            raise ValueError("not enough cells in other quadrants for all goals")
        goal_quadrant = rng.choice(choices)
        goals.append(pools[goal_quadrant].pop())

    return MAPFInstance(
        size,
        size,
        obstacles,
        [Agent(start, goal) for start, goal in zip(starts, goals)],
    )
