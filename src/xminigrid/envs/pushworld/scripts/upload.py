# Process PushWorld puzzles and save in bz2 pickle format so that it can be uploaded to huggingface
# and also used by benchmarks.py
import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Optional

import jax
import jax.numpy as jnp
import pyrallis

from xminigrid.envs.pushworld.benchmarks import load_bz2_pickle, save_bz2_pickle

PUSHWORLD_PUZZLES_PATH = os.environ.get(
    "PUSHWORLD_DATA",
    os.path.abspath("/Users/kimyoungjin/Projects/monkey/pushworld/benchmark/puzzles"),
)


@dataclass
class Config:
    # level0, level0_transformed, etc.
    level: str = "level0"
    # base, goals, all, etc.
    type: Optional[str] = None

    def __post_init__(self):
        self.is_base = self.type == "base"
        # Assert that only level0 or level0_transformed can have non-empty type
        assert self.type is None or self.level in ["level0", "level0_transformed"]


# Slight inefficiency here: we use 1-indexing by adding +1 to the coordinates.
# However, I realized this was done in the original code in order to leave space
# to add a layer of walls around the puzzle. So turns out original PushWorld was
# using 0-indexing after all.
# This mistake meant I had to do a -1 to convert to 0-indexing when decoding the puzzles
# in the training loop.
def encode_base_puzzle(puzzle_path: str):
    obj_pixels = defaultdict(set)
    with open(puzzle_path, "r") as fi:
        elems_per_row = -1
        for line_idx, line in enumerate(fi):
            y = line_idx + 1
            line_elems = line.split()
            if y == 1:
                elems_per_row = len(line_elems)
            else:
                if elems_per_row != len(line_elems):
                    raise ValueError(f"Row {y} does not have the same number of elements as the first row.")

            for x in range(1, len(line_elems) + 1):
                elem_id = line_elems[x - 1]
                if elem_id != ".":
                    obj_pixels[elem_id].add((x, y))

    encoding = jnp.hstack(
        [
            jnp.array(list(obj_pixels["A"])).flatten(),  # agent coordinates
            jnp.array(list(obj_pixels["G1"])).flatten(),  # goal coordinates
            jnp.array(list(obj_pixels["M2"])).flatten(),  # movable coordinates
            jnp.array(list(obj_pixels["M1"])).flatten(),  # movable_goal coordinates
            jnp.array(list(obj_pixels["W"])).flatten(),  # walls coordinates
        ]
    )

    return encoding


def encode_puzzle(puzzle_path: str):
    # I want to add a padding of walls so that the final puzzle size is always 10x10. For example, if the given puzzle
    # is 4x7, I want left and right paddings to be 3, and top and bottom paddings to be 2 and 1.
    # Basically center it as much as possible.
    # Therefore calculate the padding size first and then use that to start x and y ranges below at the correct positions.

    # First pass: determine dimensions
    with open(puzzle_path, "r") as fi:
        lines = fi.readlines()
        original_height = len(lines)
        original_width = len(lines[0].split())

    # Calculate padding to center the puzzle in a 10x10 grid
    target_size = 10
    x_padding_total = target_size - original_width
    y_padding_total = target_size - original_height

    # Center the puzzle by distributing padding
    x_offset = x_padding_total // 2
    y_offset = y_padding_total // 2

    # Second pass: parse with offsets applied directly
    obj_pixels = defaultdict(set)
    for line_idx, line in enumerate(lines):
        y = line_idx + y_offset  # Use 0-indexing
        line_elems = line.split()

        for x in range(len(line_elems)):  # Use 0-indexing
            elem_id = line_elems[x]
            elem_id = elem_id.lower()
            if elem_id != ".":
                obj_pixels[elem_id].add((x + x_offset, y))  # Apply x_offset directly

    # Add walls in the padded areas only
    walls = set()

    # Add walls in the top padding area
    for y in range(y_offset):
        for x in range(target_size):
            walls.add((x, y))

    # Add walls in the bottom padding area
    for y in range(y_offset + original_height, target_size):
        for x in range(target_size):
            walls.add((x, y))

    # Add walls in the left padding area (excluding corners already covered)
    for x in range(x_offset):
        for y in range(y_offset, y_offset + original_height):
            walls.add((x, y))

    # Add walls in the right padding area (excluding corners already covered)
    for x in range(x_offset + original_width, target_size):
        for y in range(y_offset, y_offset + original_height):
            walls.add((x, y))

    # Add walls to the object pixels (merge with existing walls if any)
    if "w" in obj_pixels:
        obj_pixels["w"].update(walls)
    else:
        obj_pixels["w"] = walls

    # Create fixed-length encoding with specified order and sizes
    def create_fixed_array(coords_set, max_objects):
        """
        Convert a set of (x, y) coordinates to a fixed-length flattened array.

        Args:
            coords_set: Set of (x, y) coordinate tuples
            max_objects: Maximum number of objects to include in the array

        Returns:
            JAX array of length (max_objects * 2) containing flattened coordinates,
            padded with -1s if fewer than max_objects are present.

        Example:
            coords_set = {(1, 2), (3, 4)}
            max_objects = 3
            Returns: [1, 2, 3, 4, -1, -1] (length 6)
        """
        coords_list = sorted(list(coords_set))  # Sort to ensure consistent ordering
        # Flatten coordinates and pad to required length
        flattened = []
        for i in range(max_objects):
            if i < len(coords_list):
                x, y = coords_list[i]
                flattened.extend([x, y])
            else:
                flattened.extend([-1, -1])
        return jnp.array(flattened)

    # Fixed encoding structure: a, m1, m2, m3, m4, g1, g2, w
    encoding = jnp.hstack(
        [
            create_fixed_array(obj_pixels.get("a", set()), 3),  # a: 2x3 = 6 elements
            create_fixed_array(obj_pixels.get("m1", set()), 3),  # m1: 2x3 = 6 elements
            create_fixed_array(obj_pixels.get("m2", set()), 3),  # m2: 2x3 = 6 elements
            create_fixed_array(obj_pixels.get("m3", set()), 3),  # m3: 2x3 = 6 elements
            create_fixed_array(obj_pixels.get("m4", set()), 3),  # m4: 2x3 = 6 elements
            create_fixed_array(obj_pixels.get("g1", set()), 3),  # g1: 2x3 = 6 elements
            create_fixed_array(obj_pixels.get("g2", set()), 3),  # g2: 2x3 = 6 elements
            create_fixed_array(obj_pixels.get("w", set()), 80),  # w: 2x80 = 160 elements
        ]
    )

    # Reconstruct original puzzle from obj_pixels (except now we have the wall encodings)
    # target_size = 10
    # grid = [["." for _ in range(target_size)] for _ in range(target_size)]

    # # Fill the grid with objects based on their coordinates
    # for elem_id, coords in obj_pixels.items():
    #     for x, y in coords:
    #         # Convert back to 0-indexed for grid access
    #         grid[y][x] = elem_id

    return encoding


def encode_puzzles(puzzles_path: str, encode_fn: Callable[[str], jax.Array]):
    # Open directory, loop through files inside, encode them, and return array of encodings
    encodings = []
    for file in os.listdir(puzzles_path):
        encodings.append(encode_fn(os.path.join(puzzles_path, file)))
    return jnp.array(encodings)


@pyrallis.wrap()
def main(config: Config):
    puzzles_path = os.path.join(PUSHWORLD_PUZZLES_PATH, config.level)

    if config.type is not None:
        puzzles_path = os.path.join(puzzles_path, config.type)

    if config.is_base:
        training_encodings = encode_puzzles(os.path.join(puzzles_path, "train"), encode_base_puzzle)
        test_encodings = encode_puzzles(os.path.join(puzzles_path, "test"), encode_base_puzzle)
    else:
        training_encodings = encode_puzzles(os.path.join(puzzles_path, "train"), encode_puzzle)
        test_encodings = encode_puzzles(os.path.join(puzzles_path, "test"), encode_puzzle)

    save_bz2_pickle(
        {
            "train": training_encodings,
            "test": test_encodings,
        },
        f"pushworld_{config.level}_{config.type}.pkl",
    )

    print(f"Saved to pushworld_{config.level}_{config.type}.pkl")


if __name__ == "__main__":
    main()
