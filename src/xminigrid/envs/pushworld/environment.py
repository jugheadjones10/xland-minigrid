from __future__ import annotations

import abc
from typing import Any, Generic, Literal, Optional, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from .actions import take_action
from .benchmarks import Benchmark
from .constants import LEVEL0_SIZE, NUM_ACTIONS, STEP_REWARD, SUCCESS_REWARD
from .grid import get_obs_from_puzzle
from .rgb_render import rgb_render
from .types import (
    EnvCarry,
    EnvCarryT,
    IntOrArray,
    PushWorldPuzzle,
    State,
    StateAll,
    StateT,
    StepType,
    TimeStep,
    TimeStepAll,
    TimeStepT,
)


class EnvParams(struct.PyTreeNode):
    # WARN: pytree_node=False, so you CAN NOT vmap on them!
    # You can add pytree node params, but be careful and
    # test that your code will work under jit.
    # Spoiler: probably it will not :(
    max_steps: Optional[int] = struct.field(pytree_node=False, default=100)
    render_mode: str = struct.field(pytree_node=False, default="rgb_array")
    puzzle: PushWorldPuzzle = struct.field(pytree_node=True, default=None)


EnvParamsT = TypeVar("EnvParamsT", bound="EnvParams")


class Environment(abc.ABC, Generic[EnvParamsT, TimeStepT, StateT]):
    @abc.abstractmethod
    def default_params(self, **kwargs: Any) -> EnvParamsT: ...

    def num_actions(self, params: EnvParamsT) -> int:
        return int(NUM_ACTIONS)

    # We assume level 0 only for now so we limit the shape
    def observation_shape(self, params: EnvParamsT) -> tuple[int, int, int] | dict[str, Any]:
        return LEVEL0_SIZE, LEVEL0_SIZE, 1

    # Reset logic can be different depending on whether it is single task or meta task.
    # For single task, we want to sample a new puzzle on episode reset.
    # For meta task, we want to use the same puzzle across episode resets.
    @abc.abstractmethod
    def reset(self, params: EnvParamsT, key: jax.Array) -> TimeStepT: ...

    @abc.abstractmethod
    def _generate_problem(self, params: EnvParamsT, key: jax.Array) -> StateT: ...

    # Eval reset logic can be different from normal reset, because we might want to pass in the full suite of all test
    # puzzles on every eval.
    @abc.abstractmethod
    def eval_reset(self, params: EnvParamsT, key: jax.Array) -> TimeStepT: ...

    # Why timestep + state at once, and not like in Jumanji? To be able to do autoresets in gym and envpools styles
    def step(self, params: EnvParamsT, timestep: TimeStepT, action: IntOrArray) -> TimeStepT:
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

        reward = jax.lax.select(terminated, SUCCESS_REWARD, STEP_REWARD)

        step_type = jax.lax.select(terminated | truncated, StepType.LAST, StepType.MID)
        discount = jax.lax.select(terminated, jnp.asarray(0.0), jnp.asarray(1.0))

        timestep = timestep.replace(
            state=new_state,
            step_type=step_type,
            reward=reward,
            discount=discount,
            observation=new_state.puzzle,
        )
        return timestep

    def render(self, params: EnvParamsT, timestep: TimeStepT) -> np.ndarray | str:
        if params.render_mode == "rgb_array":
            return rgb_render(np.asarray(timestep.state.puzzle))
        # elif params.render_mode == "rich_text":
        #     return text_render(timestep.state.grid, timestep.state.agent)
        else:
            raise RuntimeError("Unknown render mode. Should be one of: ['rgb_array', 'rich_text']")
