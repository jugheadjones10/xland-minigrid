from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from ..environment import Environment, EnvParams
from ..grid import get_obs_from_puzzle
from ..types import EnvCarry, State, StepType, TimeStep


class MetaTaskPushWorldEnvironment(Environment[EnvParams, EnvCarry]):
    def default_params(self, **kwargs: Any) -> EnvParams:
        params = EnvParams()
        params = params.replace(**kwargs)
        return params

    def reset(self, params: EnvParams, key: jax.Array) -> TimeStep[EnvCarry]:
        state = self._generate_problem(params, key)
        timestep = TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            observation=state.puzzle,
        )
        return timestep

    def eval_reset(self, params: EnvParams, key: jax.Array) -> TimeStep[EnvCarry]:
        return self.reset(params, key)

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State[EnvCarry]:
        obs = get_obs_from_puzzle(params.puzzle)

        state = State(
            key=key,
            step_num=jnp.asarray(0),
            puzzle=obs,
            agent_pos=(params.puzzle.agent - 1),
            goal_pos=(params.puzzle.goal - 1),
            carry=EnvCarry(),
        )
        return state
