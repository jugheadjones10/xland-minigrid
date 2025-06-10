from __future__ import annotations

from typing import Any

import jax

from .constants import LEVEL0_SIZE
from .environment import Environment, EnvParamsT
from .types import EnvCarryT, IntOrArray, State, TimeStep


class Wrapper(Environment[EnvParamsT, EnvCarryT]):
    def __init__(self, env: Environment[EnvParamsT, EnvCarryT]):
        self._env = env

    # Question: what if wrapper adds new parameters to the dataclass?
    # Solution: do this after applying the wrapper:
    #   env_params = wrapped_env.default_params(**dataclasses.asdict(original_params))
    def default_params(self, **kwargs) -> EnvParamsT:
        return self._env.default_params(**kwargs)

    def num_actions(self, params: EnvParamsT) -> int:
        return self._env.num_actions(params)

    def observation_shape(self, params: EnvParamsT) -> tuple[int, int, int] | dict[str, Any]:
        return self._env.observation_shape(params)

    def _generate_problem(self, params: EnvParamsT, key: jax.Array) -> State[EnvCarryT]:
        return self._env._generate_problem(params, key)

    def reset(self, params: EnvParamsT, key: jax.Array) -> TimeStep[EnvCarryT]:
        return self._env.reset(params, key)

    def eval_reset(self, params: EnvParamsT, key: jax.Array) -> TimeStep[EnvCarryT]:
        return self._env.eval_reset(params, key)

    def step(self, params: EnvParamsT, timestep: TimeStep[EnvCarryT], action: IntOrArray) -> TimeStep[EnvCarryT]:
        return self._env.step(params, timestep, action)

    def render(self, params: EnvParamsT, timestep: TimeStep[EnvCarryT]):
        return self._env.render(params, timestep)


# gym and gymnasium style reset (on the same step with termination)
class GymAutoResetWrapper(Wrapper):
    def __auto_reset(self, params, timestep):
        key, _ = jax.random.split(timestep.state.key)
        reset_timestep = self._env.reset(params, key)

        timestep = timestep.replace(
            state=reset_timestep.state,
            observation=reset_timestep.observation,
        )
        return timestep

    # TODO: add last_obs somewhere in the timestep? add extras like in Jumanji?
    def step(self, params, timestep, action):
        timestep = self._env.step(params, timestep, action)
        timestep = jax.lax.cond(
            timestep.last(),
            lambda: self.__auto_reset(params, timestep),
            lambda: timestep,
        )
        return timestep


# Yes, these are a bit stupid, but a tmp workaround to not write an actual system for spaces.
# Maybe, in the future, I will port the entire API to some existing one, like functional Gymnasium.
# For now, faster to do this stuff with dicts instead...
# NB: if you do not want to use this (due to the dicts as obs),
# just get needed parts from the original TimeStep and State dataclasses
class GoalObservationWrapper(Wrapper):
    def observation_shape(self, params):
        base_shape = self._env.observation_shape(params)
        if isinstance(base_shape, dict):
            assert "img" in base_shape
            obs_shape = {**base_shape, **{"goal": LEVEL0_SIZE * LEVEL0_SIZE}}
        else:
            obs_shape = {
                "img": self._env.observation_shape(params),
                "goal": LEVEL0_SIZE * LEVEL0_SIZE,
            }
        return obs_shape

    def __extend_obs(self, timestep):
        # num_classes is equal to the number of tiles in the grid
        goal_pos_index = timestep.state.goal_pos[0] * LEVEL0_SIZE + timestep.state.goal_pos[1]
        goal = jax.nn.one_hot(goal_pos_index, num_classes=LEVEL0_SIZE * LEVEL0_SIZE)
        if isinstance(timestep.observation, dict):
            assert "img" in timestep.observation
            extended_obs = {
                **timestep.observation,
                **{"goal": goal},
            }
        else:
            extended_obs = {
                "img": timestep.observation,
                "goal": goal,
            }

        timestep = timestep.replace(observation=extended_obs)
        return timestep

    def reset(self, params, key):
        timestep = self._env.reset(params, key)
        timestep = self.__extend_obs(timestep)
        return timestep

    def eval_reset(self, params, key):
        timestep = self._env.eval_reset(params, key)
        timestep = self.__extend_obs(timestep)
        return timestep

    def step(self, params, timestep, action):
        timestep = self._env.step(params, timestep, action)
        timestep = self.__extend_obs(timestep)
        return timestep
