# Process PushWorld puzzles and save in bz2 pickle format so that it can be uploaded to huggingface
# and also used by benchmarks.py
import os
import pickle
from collections import defaultdict

import jax.numpy as jnp

from ..benchmarks import load_bz2_pickle, save_bz2_pickle

PUSHWORLD_PUZZLES_PATH = os.environ.get(
    "PUSHWORLD_DATA",
    os.path.abspath("/Users/kimyoungjin/Projects/monkey/pushworld/benchmark/puzzles/level0/mini"),
    # os.path.abspath("/Users/kimyoungjin/Projects/monkey/pushworld/benchmark/puzzles/level0/mini"),
)


def encode_puzzle(puzzle_path: str):
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
                    raise ValueError(f"Row {y} does not have the same number of elements as " "the first row.")

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


def encode_puzzles(puzzles_path: str):
    # Open directory, loop through files inside, encode them, and return array of encodings
    encodings = []
    for file in os.listdir(puzzles_path):
        encodings.append(encode_puzzle(os.path.join(puzzles_path, file)))
    return jnp.array(encodings)


if __name__ == "__main__":
    training_puzzles_path = os.path.join(PUSHWORLD_PUZZLES_PATH, "train")
    test_puzzles_path = os.path.join(PUSHWORLD_PUZZLES_PATH, "test")

    training_encodings = encode_puzzles(training_puzzles_path)
    test_encodings = encode_puzzles(test_puzzles_path)

    # Save encodings to file
    save_bz2_pickle(
        {
            "train": training_encodings,
            "test": test_encodings,
        },
        "pushworld_level0_mini.pkl",
    )
