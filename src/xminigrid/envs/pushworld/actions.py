import jax
import jax.numpy as jnp
from typing_extensions import TypeAlias

from .constants import LEVEL0_SIZE, Tiles
from .types import GridState, IntOrArray, PushWorldPuzzle

# Movement actions should have the following logic:
# Displace current agent coordinates in the correct direction depending on what action it is.
# This will give us the coordinates of the new agent if the action is taken.

# 1. Check if ths new agent coordinate is empty. If yes, move the agent there.
# 2. Check if there is a wall at the new agent coordinate. If yes, do nothing.
# 3. Check if there is a movable object at the new agent coordinate.
#    - If yes, calculate the new coordinates of the movable object.
#    - Check if that new coordinate is empty. If yes, move both agent and movable object in that direction
#    - If there is a wall at the new movable object coordinate, do nothing.
#    - If there is another movable object at the new movable object coordinate, repeat the above 3 steps
#      one more time.

# Keep in mind that all our state is represented by GridState,
# which is just a 5x5 jax array where the number represents the object at that position.
# Tile identity is given by:
# class Tiles(struct.PyTreeNode):
#     EMPTY: int = struct.field(pytree_node=False, default=0)
#     AGENT: int = struct.field(pytree_node=False, default=1)
#     GOAL: int = struct.field(pytree_node=False, default=2)
#     MOVABLE: int = struct.field(pytree_node=False, default=3)
#     MOVABLE_GOAL: int = struct.field(pytree_node=False, default=4)
#     WALL: int = struct.field(pytree_node=False, default=5)

# The second item returns the new position of the agent
ActionOutput: TypeAlias = tuple[GridState, jax.Array, jax.Array]


def _apply_move(
    grid: GridState,
    agent_pos: jax.Array,
    goal_pos: jax.Array,
    delta: jax.Array,
) -> ActionOutput:
    zero = jnp.array((0, 0))
    max_xy = jnp.array((LEVEL0_SIZE - 1, LEVEL0_SIZE - 1))

    pos1 = jnp.clip(agent_pos + delta, zero, max_xy)
    tile1 = grid[pos1[1], pos1[0]]

    # CASE A: move into empty cell → no goal reached
    def _move_agent(g):
        g = g.at[agent_pos[1], agent_pos[0]].set(Tiles.EMPTY)
        g = g.at[pos1[1], pos1[0]].set(Tiles.AGENT)
        return g, pos1, jnp.array(False)

    # CASE B: attempt to push a box (either MOVABLE or MOVABLE_GOAL)
    def _push_box(g):
        pos2 = jnp.clip(pos1 + delta, zero, max_xy)
        tile2 = g[pos2[1], pos2[0]]

        def _do_push_one(h):
            # only succeed if can_push1 was True
            reached = (tile1 == Tiles.MOVABLE_GOAL) & jnp.all(pos2 == goal_pos)
            # preserve box type (MOVABLE or MOVABLE_GOAL)
            # jax.debug.print(h)
            h = h.at[pos2[1], pos2[0]].set(tile1)
            h = h.at[pos1[1], pos1[0]].set(Tiles.AGENT)
            h = h.at[agent_pos[1], agent_pos[0]].set(Tiles.EMPTY)
            # jax.debug.print(h)
            return h, pos1, reached

        # 2-box chain push?
        def _try_push_two(h):
            pos3 = jnp.clip(pos2 + delta, zero, max_xy)
            tile3 = h[pos3[1], pos3[0]]

            def _do_push_two(k):
                reached = (tile1 == Tiles.MOVABLE_GOAL) & jnp.all(pos3 == goal_pos)
                # move second box (tile1) → pos3
                k = k.at[pos3[1], pos3[0]].set(tile1)
                # move first box (tile1) → pos2
                k = k.at[pos2[1], pos2[0]].set(tile1)
                # move agent → pos1
                k = k.at[pos1[1], pos1[0]].set(Tiles.AGENT)
                # clear old agent
                k = k.at[agent_pos[1], agent_pos[0]].set(Tiles.EMPTY)
                return k, pos1, reached

            return jax.lax.cond(tile3 == Tiles.EMPTY, _do_push_two, lambda k: (k, agent_pos, jnp.array(False)), h)

        # choose 1-box vs chain-push vs no-op
        is_box = (tile2 == Tiles.MOVABLE) | (tile2 == Tiles.MOVABLE_GOAL)
        return jax.lax.cond(
            tile2 == Tiles.EMPTY,
            _do_push_one,
            lambda h: jax.lax.cond(is_box, _try_push_two, lambda k: (k, agent_pos, jnp.array(False)), h),
            g,
        )

    # TOP LEVEL: empty → move, box → push, wall/goal on pos1 → no-op
    is_box = (tile1 == Tiles.MOVABLE) | (tile1 == Tiles.MOVABLE_GOAL)
    return jax.lax.cond(
        tile1 == Tiles.EMPTY,
        _move_agent,
        lambda h: jax.lax.cond(is_box, _push_box, lambda k: (k, agent_pos, jnp.array(False)), h),
        grid,
    )


# Ok, I'm sorry but change implementation yet again so that env params or state holds the goal position.
# We can't have goal on grid because there can't be two overlapping items on the grid.
# Remember to remove the GOAL type TILE from the grid creation logic.


def move_left(grid: GridState, agent_pos: jax.Array, goal_pos: jax.Array) -> ActionOutput:
    return _apply_move(grid, agent_pos, goal_pos, jnp.array((-1, 0)))


def move_right(grid: GridState, agent_pos: jax.Array, goal_pos: jax.Array) -> ActionOutput:
    return _apply_move(grid, agent_pos, goal_pos, jnp.array((1, 0)))


def move_up(grid: GridState, agent_pos: jax.Array, goal_pos: jax.Array) -> ActionOutput:
    return _apply_move(grid, agent_pos, goal_pos, jnp.array((0, -1)))


def move_down(grid: GridState, agent_pos: jax.Array, goal_pos: jax.Array) -> ActionOutput:
    return _apply_move(grid, agent_pos, goal_pos, jnp.array((0, 1)))


def take_action(grid: GridState, agent_pos: jax.Array, goal_pos: jax.Array, action: IntOrArray) -> ActionOutput:
    actions = (
        lambda: move_up(grid, agent_pos, goal_pos),  # 0
        lambda: move_right(grid, agent_pos, goal_pos),  # 1
        lambda: move_down(grid, agent_pos, goal_pos),  # 2
        lambda: move_left(grid, agent_pos, goal_pos),  # 3
    )
    return jax.lax.switch(action, actions)
