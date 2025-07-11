import jax.numpy as jnp
from flax import struct

NUM_ACTIONS = 4

# GRID: [tile, color]
NUM_TILES = 5
LEVEL0_SIZE = 5

LEVEL0_ALL_SIZE = 10
LEVEL0_NUM_CHANNELS = 8

SUCCESS_REWARD = 10.0
STEP_REWARD = -0.01


# enums, kinda...
class Tiles(struct.PyTreeNode):
    EMPTY: int = struct.field(pytree_node=False, default=0)
    AGENT: int = struct.field(pytree_node=False, default=1)
    MOVABLE: int = struct.field(pytree_node=False, default=2)
    MOVABLE_GOAL: int = struct.field(pytree_node=False, default=3)
    WALL: int = struct.field(pytree_node=False, default=4)


# DIRECTIONS = jnp.array(
#     (
#         (-1, 0),  # Up
#         (0, 1),  # Right
#         (1, 0),  # Down
#         (0, -1),  # Left
#     )
# )
