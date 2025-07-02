from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from ..actions_all import check_goal, num_goals_reached, take_action_all
from ..benchmarks import BenchmarkAll
from ..constants import LEVEL0_ALL_SIZE, LEVEL0_NUM_CHANNELS, STEP_REWARD, SUCCESS_REWARD
from ..environment import Environment, EnvParams
from ..grid import get_obs_from_puzzle
from ..observation import get_obs_from_puzzle_all
from ..types import (
    EnvCarry,
    IntOrArray,
    PushWorldPuzzleAll,
    State,
    StateAll,
    StepType,
    TimeStep,
    TimeStepAll,
)


class SingleTaskPushWorldEnvParamsAll(EnvParams):
    puzzle: PushWorldPuzzleAll = struct.field(pytree_node=True, default=None)
    benchmark: BenchmarkAll = struct.field(pytree_node=True, default=None)


class SingleTaskPushWorldEnvironmentAll(
    Environment[SingleTaskPushWorldEnvParamsAll, TimeStepAll[EnvCarry], StateAll[EnvCarry]]
):
    def default_params(self, **kwargs: Any) -> SingleTaskPushWorldEnvParamsAll:
        params = SingleTaskPushWorldEnvParamsAll()
        params = params.replace(**kwargs)
        return params

    # We assume level 0 only for now so we limit the shape
    def observation_shape(self, params: SingleTaskPushWorldEnvParamsAll) -> tuple[int, int, int] | dict[str, Any]:
        return (
            LEVEL0_ALL_SIZE,
            LEVEL0_ALL_SIZE,
            LEVEL0_NUM_CHANNELS,
        )

    def reset(self, params: SingleTaskPushWorldEnvParamsAll, key: jax.Array) -> TimeStepAll[EnvCarry]:
        puzzle = params.benchmark.sample_puzzle(key, "train")
        params = params.replace(puzzle=puzzle)
        state = self._generate_problem(params, key)
        # Generate observation
        observation = get_obs_from_puzzle_all(params.puzzle, state)
        timestep = TimeStepAll(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=observation,
        )
        return timestep

    def eval_reset(self, params: SingleTaskPushWorldEnvParamsAll, key: jax.Array) -> TimeStepAll[EnvCarry]:
        state = self._generate_problem(params, key)
        # Generate observation
        observation = get_obs_from_puzzle_all(params.puzzle, state)
        timestep = TimeStepAll(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=observation,
        )
        return timestep

    def step(
        self, params: SingleTaskPushWorldEnvParamsAll, timestep: TimeStepAll[EnvCarry], action: IntOrArray
    ) -> TimeStepAll[EnvCarry]:
        # Input to take_action_all:
        # timestep.observation, action, timestep.state
        new_state = take_action_all(timestep.observation, timestep.state, action)

        new_state = new_state.replace(
            step_num=timestep.state.step_num + 1,
        )
        new_observation = get_obs_from_puzzle_all(new_state.puzzle, new_state)

        # Calculate reward based on goal progress
        prev_goals_reached = num_goals_reached(timestep.observation)
        new_goals_reached = num_goals_reached(new_observation)

        terminated = check_goal(new_observation, new_state)

        assert params.max_steps is not None
        truncated = jnp.equal(new_state.step_num, params.max_steps)

        reward = jax.lax.select(terminated, SUCCESS_REWARD, new_goals_reached - prev_goals_reached + STEP_REWARD)

        step_type = jax.lax.select(terminated | truncated, StepType.LAST, StepType.MID)
        discount = jax.lax.select(terminated, jnp.asarray(0.0), jnp.asarray(1.0))

        timestep = timestep.replace(
            state=new_state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=new_observation,
        )
        return timestep

    def _generate_problem(self, params: SingleTaskPushWorldEnvParamsAll, key: jax.Array) -> StateAll[EnvCarry]:
        state = StateAll(
            key=key,
            step_num=jnp.asarray(0),
            puzzle=params.puzzle,
            a=params.puzzle.a,
            m1=params.puzzle.m1,
            m2=params.puzzle.m2,
            m3=params.puzzle.m3,
            m4=params.puzzle.m4,
            carry=EnvCarry(),
        )
        return state
