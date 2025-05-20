from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from ...core.actions import take_action
from ...core.constants import TILES_REGISTRY, Colors, Tiles
from ...core.goals import AgentOnTileGoal, check_goal
from ...core.grid import room, sample_coordinates, sample_direction
from ...core.observation import transparent_field_of_view, transparent_field_of_view_hidden_goal
from ...core.rules import EmptyRule, check_rule
from ...environment import Environment, EnvParams, EnvParamsT
from ...types import AgentState, EnvCarry, EnvCarryT, IntOrArray, State, StepType, TimeStep

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

    def reset(self, params: EmptyGoalRandomEnvParams, key: jax.Array) -> TimeStep[EnvCarry]:
        state = self._generate_problem(params, key)
        timestep = TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=transparent_field_of_view_hidden_goal(
                state.grid, state.agent, params.view_size, params.view_size
            ),
        )
        return timestep

    # Why timestep + state at once, and not like in Jumanji? To be able to do autoresets in gym and envpools styles
    def step(self, params: EnvParamsT, timestep: TimeStep[EnvCarryT], action: IntOrArray) -> TimeStep[EnvCarryT]:
        new_grid, new_agent, changed_position = take_action(timestep.state.grid, timestep.state.agent, action)
        new_grid, new_agent = check_rule(timestep.state.rule_encoding, new_grid, new_agent, action, changed_position)

        new_state = timestep.state.replace(
            grid=new_grid,
            agent=new_agent,
            step_num=timestep.state.step_num + 1,
        )
        new_observation = transparent_field_of_view_hidden_goal(
            new_state.grid, new_state.agent, params.view_size, params.view_size
        )

        # checking for termination or truncation, choosing step type
        terminated = check_goal(new_state.goal_encoding, new_state.grid, new_state.agent, action, changed_position)

        assert params.max_steps is not None
        truncated = jnp.equal(new_state.step_num, params.max_steps)

        reward = jax.lax.select(terminated, 1.0 - 0.9 * (new_state.step_num / params.max_steps), 0.0)

        step_type = jax.lax.select(terminated | truncated, StepType.LAST, StepType.MID)
        discount = jax.lax.select(terminated, jnp.asarray(0.0), jnp.asarray(1.0))

        timestep = TimeStep(
            state=new_state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=new_observation,
        )
        return timestep


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
