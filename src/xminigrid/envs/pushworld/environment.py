from __future__ import annotations

import abc
from typing import Any, Generic, Literal, Optional, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from .actions import take_action
from .benchmarks import Benchmark
from .constants import LEVEL0_SIZE, NUM_ACTIONS
from .grid import get_obs_from_puzzle
from .rgb_render import rgb_render
from .types import EnvCarry, EnvCarryT, IntOrArray, PushWorldPuzzle, State, StepType, TimeStep


class EnvParams(struct.PyTreeNode):
    # WARN: pytree_node=False, so you CAN NOT vmap on them!
    # You can add pytree node params, but be careful and
    # test that your code will work under jit.
    # Spoiler: probably it will not :(
    max_steps: Optional[int] = struct.field(pytree_node=False, default=100)
    render_mode: str = struct.field(pytree_node=False, default="rgb_array")

    # rng key used to choose random level 0 puzzle
    puzzle: PushWorldPuzzle = struct.field(pytree_node=True, default=None)


EnvParamsT = TypeVar("EnvParamsT", bound="EnvParams")


class Environment(abc.ABC, Generic[EnvParamsT, EnvCarryT]):
    @abc.abstractmethod
    def default_params(self, **kwargs: Any) -> EnvParamsT: ...

    def num_actions(self, params: EnvParamsT) -> int:
        return int(NUM_ACTIONS)

    # We assume level 0 only for now so we limit the shape
    def observation_shape(self, params: EnvParamsT) -> tuple[int, int, int] | dict[str, Any]:
        return LEVEL0_SIZE, LEVEL0_SIZE, 1

    @abc.abstractmethod
    def _generate_problem(self, params: EnvParamsT, key: jax.Array) -> State[EnvCarryT]: ...

    def reset(self, params: EnvParamsT, key: jax.Array) -> TimeStep[EnvCarryT]:
        state = self._generate_problem(params, key)
        timestep = TimeStep(
            state=state,
            step_type=StepType.FIRST,
            reward=jnp.asarray(0.0),
            discount=jnp.asarray(1.0),
            # A bit redundant but we want to stick to the original API as much as possible
            observation=state.puzzle,
        )
        return timestep

    # Why timestep + state at once, and not like in Jumanji? To be able to do autoresets in gym and envpools styles
    def step(self, params: EnvParamsT, timestep: TimeStep[EnvCarryT], action: IntOrArray) -> TimeStep[EnvCarryT]:
        new_grid, changed_position, goal_reached = take_action(
            timestep.state.puzzle, timestep.state.agent_pos, timestep.state.goal_pos, action
        )

        new_state = timestep.state.replace(
            puzzle=new_grid,
            step_num=timestep.state.step_num + 1,
            agent_pos=changed_position,
        )

        # checking for termination or truncation, choosing step type
        terminated = goal_reached

        assert params.max_steps is not None
        truncated = jnp.equal(new_state.step_num, params.max_steps)

        reward = jax.lax.select(terminated, 10.0, -0.01)

        step_type = jax.lax.select(terminated | truncated, StepType.LAST, StepType.MID)
        discount = jax.lax.select(terminated, jnp.asarray(0.0), jnp.asarray(1.0))

        timestep = TimeStep(
            state=new_state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=new_state.puzzle,
        )
        return timestep

    def render(self, params: EnvParamsT, timestep: TimeStep[EnvCarryT]) -> np.ndarray | str:
        if params.render_mode == "rgb_array":
            return rgb_render(np.asarray(timestep.state.puzzle))
        # elif params.render_mode == "rich_text":
        #     return text_render(timestep.state.grid, timestep.state.agent)
        else:
            raise RuntimeError("Unknown render mode. Should be one of: ['rgb_array', 'rich_text']")


class PushWorldEnvironment(Environment[EnvParams, EnvCarry]):
    def default_params(self, **kwargs: Any) -> EnvParams:
        params = EnvParams()
        params = params.replace(**kwargs)
        return params

    def _generate_problem(self, params: EnvParams, key: jax.Array) -> State[EnvCarry]:
        # Create a new state, copy PushWorld from params into it
        # TODO: Later on, check if we even need to copy or whether we can use the same puzzle
        # due to how jax works
        # puzzle_copy = jtu.tree_map(lambda x: x, params.puzzle)

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


class PushWorldSingleTaskEnvParams(EnvParams):
    benchmark: Benchmark = struct.field(pytree_node=True, default=None)
    type: Literal["train", "test"] = struct.field(pytree_node=False, default="train")
    eval_puzzle_ids: jax.Array = struct.field(pytree_node=True, default=jnp.array([]))


class PushWorldSingleTaskEnvironment(Environment[PushWorldSingleTaskEnvParams, EnvCarry]):
    def default_params(self, **kwargs: Any) -> PushWorldSingleTaskEnvParams:
        params = PushWorldSingleTaskEnvParams()
        params = params.replace(**kwargs)
        return params

    def _generate_problem(self, params: PushWorldSingleTaskEnvParams, key: jax.Array) -> State[EnvCarry]:
        puzzle = params.benchmark.sample_puzzle(key, params.type)
        obs = get_obs_from_puzzle(puzzle)

        state = State(
            key=key,
            step_num=jnp.asarray(0),
            puzzle=obs,
            agent_pos=(puzzle.agent - 1),
            goal_pos=(puzzle.goal - 1),
            carry=EnvCarry(),
        )
        return state
