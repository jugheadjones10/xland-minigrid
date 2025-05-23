from __future__ import annotations

import bz2
import os
import pickle
import urllib.request
from typing import Literal

import jax
import jax.numpy as jnp
from flax import struct
from tqdm.auto import tqdm

from .types import PushWorldPuzzle

HF_REPO_ID = os.environ.get("PUSHWORLD_HF_REPO_ID", "feynmaniac/pushworld")
DATA_PATH = os.environ.get("PUSHWORLD_DATA", os.path.expanduser("~/.pushworld"))

NAME2HFFILENAME = {
    "level0_transformed_base": "pushworld_level0_transformed_base.pkl",
    "level0_mini": "pushworld_level0_mini.pkl",
}


# jit compatible sampling and indexing!
# You can implement your custom curriculums based on this class.
class Benchmark(struct.PyTreeNode):
    train_puzzles: jax.Array
    test_puzzles: jax.Array

    def num_train_puzzles(self) -> int:
        return len(self.train_puzzles)

    def num_test_puzzles(self) -> int:
        return len(self.test_puzzles)

    def get_puzzle(self, puzzle_id: int | jax.Array, type: Literal["train", "test"] = "train") -> PushWorldPuzzle:
        puzzle = jax.lax.cond(
            type == "train",
            lambda: jax.lax.dynamic_index_in_dim(self.train_puzzles, puzzle_id, keepdims=False),
            lambda: jax.lax.dynamic_index_in_dim(self.test_puzzles, puzzle_id, keepdims=False),
        )
        return PushWorldPuzzle(
            agent=puzzle[:2],
            goal=puzzle[2:4],
            movable=puzzle[4:6],
            movable_goal=puzzle[6:8],
            walls=puzzle[8:],
        )

    def sample_puzzle(self, key: jax.Array, type: Literal["train", "test"] = "train") -> PushWorldPuzzle:
        # Split the key for random number generation
        key, _key = jax.random.split(key)

        # Use jax.lax.cond for control flow
        puzzle_id = jax.lax.cond(
            type == "train",
            lambda k: jax.random.randint(k, shape=(), minval=0, maxval=self.num_train_puzzles()),
            lambda k: jax.random.randint(k, shape=(), minval=0, maxval=self.num_test_puzzles()),
            _key,
        )
        return self.get_puzzle(puzzle_id, type)

    # def shuffle(self, key: jax.Array) -> Benchmark:
    #     idxs = jax.random.permutation(key, jnp.arange(len(self.num_rules)))
    #     return jtu.tree_map(lambda a: a[idxs], self)

    # def split(self, prop: float) -> tuple[Benchmark, Benchmark]:
    #     idx = round(len(self.num_rules) * prop)
    #     bench1 = jtu.tree_map(lambda a: a[:idx], self)
    #     bench2 = jtu.tree_map(lambda a: a[idx:], self)
    #     return bench1, bench2

    # def filter_split(self, fn: Callable[[jax.Array, jax.Array], bool]) -> tuple[Benchmark, Benchmark]:
    #     # fn(single_goal, single_rules) -> bool
    #     mask = jax.vmap(fn)(self.goals, self.rules)
    #     bench1 = jtu.tree_map(lambda a: a[mask], self)
    #     bench2 = jtu.tree_map(lambda a: a[~mask], self)
    #     return bench1, bench2


def load_benchmark_from_path(path: str) -> Benchmark:
    puzzle_dict = load_bz2_pickle(path)
    benchmark = Benchmark(
        train_puzzles=puzzle_dict["train"],
        test_puzzles=puzzle_dict["test"],
    )
    return benchmark


def load_benchmark(name: str) -> Benchmark:
    if name not in NAME2HFFILENAME:
        raise RuntimeError(f"Unknown benchmark. Registered: {registered_benchmarks()}")

    os.makedirs(DATA_PATH, exist_ok=True)

    path = os.path.join(DATA_PATH, NAME2HFFILENAME[name])
    if not os.path.exists(path):
        _download_from_hf(HF_REPO_ID, NAME2HFFILENAME[name])

    return load_benchmark_from_path(path)


def registered_benchmarks() -> tuple[str, ...]:
    return tuple(NAME2HFFILENAME.keys())


def _download_from_hf(repo_id: str, filename: str) -> None:
    dataset_url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"

    save_path = os.path.join(DATA_PATH, filename)
    print(f"Downloading benchmark data: {dataset_url} to {DATA_PATH}")

    with tqdm(unit="B", unit_scale=True, miniters=1, desc="Progress") as t:

        def progress_hook(block_num=1, block_size=1, total_size=None):
            if total_size is not None:
                t.total = total_size
            t.update(block_num * block_size - t.n)

        urllib.request.urlretrieve(dataset_url, save_path, reporthook=progress_hook)

    if not os.path.exists(os.path.join(DATA_PATH, filename)):
        raise IOError(f"Failed to download benchmark data from {dataset_url}")


def save_bz2_pickle(puzzles: dict[str, jax.Array], path: str, protocol: int = -1) -> None:
    with bz2.open(path, "wb") as f:
        pickle.dump(puzzles, f, protocol=protocol)


def load_bz2_pickle(path: str) -> dict[str, jax.Array]:
    with bz2.open(path, "rb") as f:
        puzzles = pickle.load(f)
    return puzzles
