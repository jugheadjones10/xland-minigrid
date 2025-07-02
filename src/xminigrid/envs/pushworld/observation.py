import jax
import jax.numpy as jnp

from .constants import LEVEL0_ALL_SIZE
from .types import PushWorldPuzzleAll, StateAll


def create_channel(coords_array: jax.Array) -> jax.Array:
    """Create a single channel (2D grid) with 1s at coordinate locations."""
    # Reshape flattened array to (n_objects, 2) for (x, y) pairs
    coords = coords_array.reshape(-1, 2)

    # Create mask for valid coordinates (not padding -1s)
    valid_mask = (coords[:, 0] != -1) & (coords[:, 1] != -1)

    # Create indices - use jnp.where to handle invalid coordinates
    y_indices = jnp.where(valid_mask, coords[:, 1], 0)
    x_indices = jnp.where(valid_mask, coords[:, 0], 0)

    values = valid_mask.astype(jnp.int32)

    # Use scatter_add to place all values at once
    channel = jnp.zeros((LEVEL0_ALL_SIZE, LEVEL0_ALL_SIZE), dtype=jnp.int32)
    channel = channel.at[y_indices, x_indices].add(values)

    return channel


def get_obs_from_puzzle_all(puzzle: PushWorldPuzzleAll, state: StateAll) -> jax.Array:
    # Observation should be a stack of 8 channels, where each channel is a 2D array of 0s or 1s.
    # There is a 1 if the corresponding object is at that location.
    # The objects are: a, m1, m2, m3, m4, g1, g2, w

    # Create channels for each object type in order: a, m1, m2, m3, m4, g1, g2, w
    channels = []
    channels.append(create_channel(state.a))  # agent
    channels.append(create_channel(state.m1))  # movable 1
    channels.append(create_channel(state.m2))  # movable 2
    channels.append(create_channel(state.m3))  # movable 3
    channels.append(create_channel(state.m4))  # movable 4
    channels.append(create_channel(puzzle.g1))  # goal 1
    channels.append(create_channel(puzzle.g2))  # goal 2
    channels.append(create_channel(puzzle.w))  # walls

    # Stack channels to create final observation: (height, width, channels)
    observation = jnp.stack(channels, axis=-1)

    return observation
