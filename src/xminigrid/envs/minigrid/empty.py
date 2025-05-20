from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from ...core.constants import TILES_REGISTRY, Colors, Tiles
from ...core.goals import AgentOnTileGoal
from ...core.grid import room, sample_coordinates, sample_direction
from ...core.rules import EmptyRule
from ...environment import Environment, EnvParams
from ...types import AgentState, EnvCarry, State

# goals and rules are hardcoded for minigrid envs
_goal_encoding = AgentOnTileGoal(tile=TILES_REGISTRY[Tiles.GOAL, Colors.GREEN]).encode()
_rule_encoding = EmptyRule().encode()[None, ...]


class Empty(Environment[EnvParams, EnvCarry]):
    def default_params(self, **kwargs) -> EnvParams:
        params = EnvParams(height=9, width=9)
        params = params.replace(**kwargs)

        if params.max_steps is None:
            # formula directly taken from MiniGrid
            params = params.replace(max_steps=4 * (params.height * params.width))
        return params

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State[EnvCarry]:
        grid = room(params.height, params.width)

        grid = grid.at[params.height - 2, params.width - 2].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])
        agent = AgentState(position=jnp.array((1, 1)), direction=jnp.asarray(1))

        state = State(
            key=key,
            step_num=jnp.asarray(0),
            grid=grid,
            agent=agent,
            goal_encoding=_goal_encoding,
            rule_encoding=_rule_encoding,
            carry=EnvCarry(),
        )
        return state


class EmptyGoalRandomEnvParams(EnvParams):
    puzzle_key: jax.Array = struct.field(pytree_node=True, default_factory=lambda: jnp.asarray(0))


class EmptyGoalRandom(Environment[EmptyGoalRandomEnvParams, EnvCarry]):
    def default_params(self, **kwargs) -> EmptyGoalRandomEnvParams:
        params = EmptyGoalRandomEnvParams(height=9, width=9)
        params = params.replace(**kwargs)

        if params.max_steps is None:
            # formula directly taken from MiniGrid
            params = params.replace(max_steps=4 * (params.height * params.width))
        return params

    def _generate_problem(self, params: EmptyGoalRandomEnvParams, key: jax.Array) -> State[EnvCarry]:
        grid = room(params.height, params.width)

        mask = jnp.ones((grid.shape[0], grid.shape[1]), dtype=jnp.bool_)
        # We mask this because agent starts at (1, 1)
        mask = mask.at[1, 1].set(False)
        # We always use the same puzzle key for the current puzzle since we are in meta-RL setting
        pos = sample_coordinates(params.puzzle_key, grid, num=1, mask=mask).squeeze()
        grid = grid.at[pos[0], pos[1]].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])

        agent = AgentState(
            position=jnp.array((1, 1)),
            direction=jnp.asarray(1),
        )

        state = State(
            key=key,
            step_num=jnp.asarray(0),
            grid=grid,
            agent=agent,
            goal_encoding=_goal_encoding,
            rule_encoding=_rule_encoding,
            carry=EnvCarry(),
        )
        return state


class EmptyRandom(Environment[EnvParams, EnvCarry]):
    def default_params(self, **kwargs) -> EnvParams:
        params = EnvParams(height=9, width=9)
        params = params.replace(**kwargs)

        if params.max_steps is None:
            # formula directly taken from MiniGrid
            params = params.replace(max_steps=4 * (params.height * params.width))
        return params

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State[EnvCarry]:
        key, pos_key, dir_key = jax.random.split(key, num=3)

        grid = room(params.height, params.width)
        grid = grid.at[params.height - 2, params.width - 2].set(TILES_REGISTRY[Tiles.GOAL, Colors.GREEN])

        agent = AgentState(
            position=sample_coordinates(pos_key, grid, num=1).squeeze(), direction=sample_direction(dir_key)
        )
        state = State(
            key=key,
            step_num=jnp.asarray(0),
            grid=grid,
            agent=agent,
            goal_encoding=_goal_encoding,
            rule_encoding=_rule_encoding,
            carry=EnvCarry(),
        )
        return state
