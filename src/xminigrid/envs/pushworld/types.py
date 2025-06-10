from __future__ import annotations

from typing import Generic, TypeVar, Union

import jax
import jax.numpy as jnp
from flax import struct
from typing_extensions import TypeAlias


class PushWorldPuzzle(struct.PyTreeNode):
    id: jax.Array
    agent: jax.Array
    goal: jax.Array
    movable: jax.Array
    movable_goal: jax.Array
    walls: jax.Array


class EnvCarry(struct.PyTreeNode): ...


EnvCarryT = TypeVar("EnvCarryT")
IntOrArray: TypeAlias = Union[int, jax.Array]
GridState: TypeAlias = jax.Array


# Originally struct.PyTreeNode came first, but we ran into this issue:
# AttributeError: type object 'TimeStep' has no attribute '__parameters__'
class State(Generic[EnvCarryT], struct.PyTreeNode):
    key: jax.Array
    step_num: jax.Array

    puzzle: jax.Array
    agent_pos: jax.Array
    goal_pos: jax.Array

    carry: EnvCarryT


class StepType(jnp.uint8):
    FIRST: jax.Array = jnp.asarray(0, dtype=jnp.uint8)
    MID: jax.Array = jnp.asarray(1, dtype=jnp.uint8)
    LAST: jax.Array = jnp.asarray(2, dtype=jnp.uint8)


# Originally struct.PyTreeNode came first, but we ran into this issue:
# AttributeError: type object 'TimeStep' has no attribute '__parameters__'
class TimeStep(Generic[EnvCarryT], struct.PyTreeNode):
    state: State[EnvCarryT]
    step_type: StepType
    reward: jax.Array
    discount: jax.Array
    observation: jax.Array

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST
