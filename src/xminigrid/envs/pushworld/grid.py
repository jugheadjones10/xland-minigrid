import jax
import jax.numpy as jnp

from .constants import LEVEL0_SIZE, Tiles
from .types import GridState, IntOrArray, PushWorldPuzzle


def empty_world(height: IntOrArray, width: IntOrArray) -> GridState:
    grid = jnp.zeros((height, width), dtype=jnp.uint8)
    grid = grid.at[:, :].set(Tiles.EMPTY)
    return grid


def get_obs_from_puzzle(puzzle: PushWorldPuzzle) -> GridState:
    grid = empty_world(LEVEL0_SIZE, LEVEL0_SIZE)

    # set agent at row=y_a-1, col=x_a-1
    grid = grid.at[puzzle.agent[1] - 1, puzzle.agent[0] - 1].set(Tiles.AGENT)
    grid = grid.at[puzzle.movable[1] - 1, puzzle.movable[0] - 1].set(Tiles.MOVABLE)
    grid = grid.at[puzzle.movable_goal[1] - 1, puzzle.movable_goal[0] - 1].set(Tiles.MOVABLE_GOAL)

    # walls
    grid = grid.at[puzzle.walls[1] - 1, puzzle.walls[0] - 1].set(Tiles.WALL)
    grid = grid.at[puzzle.walls[3] - 1, puzzle.walls[2] - 1].set(Tiles.WALL)
    grid = grid.at[puzzle.walls[5] - 1, puzzle.walls[4] - 1].set(Tiles.WALL)

    return grid
