from __future__ import annotations

import abc
import bz2
import os
import pickle
import urllib.request
from typing import Generic, Literal

import jax
import jax.numpy as jnp
from flax import struct
from tqdm.auto import tqdm

from .constants import LEVEL0_ALL_SIZE
from .types import ADRParams, PushWorldPuzzle, PushWorldPuzzleAll, PuzzleT

HF_REPO_ID = os.environ.get("PUSHWORLD_HF_REPO_ID", "feynmaniac/pushworld")
DATA_PATH = os.environ.get("PUSHWORLD_DATA", os.path.expanduser("~/.pushworld"))

NAME2HFFILENAME = {
    "level0_transformed_all": "pushworld_level0_transformed_all.pkl",
    "level0_transformed_base": "pushworld_level0_transformed_base.pkl",
    "level0_mini": "pushworld_level0_mini.pkl",
}


class BenchmarkBase(Generic[PuzzleT], struct.PyTreeNode):
    train_puzzles: jax.Array
    test_puzzles: jax.Array

    def num_train_puzzles(self) -> int:
        return len(self.train_puzzles)

    def num_test_puzzles(self) -> int:
        return len(self.test_puzzles)

    @abc.abstractmethod
    def get_puzzle(self, puzzle_id: jax.Array, type: Literal["train", "test"] = "train") -> PuzzleT: ...

    def get_test_puzzles(self) -> PuzzleT:
        return jax.vmap(self.get_puzzle, in_axes=(0, None))(jnp.arange(self.num_test_puzzles()), "test")

    def get_train_puzzles(self) -> PuzzleT:
        return jax.vmap(self.get_puzzle, in_axes=(0, None))(jnp.arange(self.num_train_puzzles()), "train")

    def sample_puzzle(self, key: jax.Array, type: Literal["train", "test"] = "train") -> PuzzleT:
        key, _key = jax.random.split(key)
        puzzle_id = jax.lax.cond(
            type == "train",
            lambda k: jax.random.randint(k, shape=(), minval=0, maxval=self.num_train_puzzles()),
            lambda k: jax.random.randint(k, shape=(), minval=0, maxval=self.num_test_puzzles()),
            _key,
        )
        return self.get_puzzle(puzzle_id, type)


# jit compatible sampling and indexing!
# You can implement your custom curriculums based on this class.
class Benchmark(BenchmarkBase[PushWorldPuzzle]):
    def get_puzzle(self, puzzle_id: jax.Array, type: Literal["train", "test"] = "train") -> PushWorldPuzzle:
        puzzle = jax.lax.cond(
            type == "train",
            lambda: jax.lax.dynamic_index_in_dim(self.train_puzzles, puzzle_id, keepdims=False),
            lambda: jax.lax.dynamic_index_in_dim(self.test_puzzles, puzzle_id, keepdims=False),
        )
        return PushWorldPuzzle(
            id=puzzle_id,
            agent=puzzle[:2],
            goal=puzzle[2:4],
            movable=puzzle[4:6],
            movable_goal=puzzle[6:8],
            walls=puzzle[8:],
        )


class BenchmarkAll(BenchmarkBase[PushWorldPuzzleAll]):
    def get_puzzle(self, puzzle_id: jax.Array, type: Literal["train", "test"] = "train") -> PushWorldPuzzleAll:
        puzzle = jax.lax.cond(
            type == "train",
            lambda: jax.lax.dynamic_index_in_dim(self.train_puzzles, puzzle_id, keepdims=False),
            lambda: jax.lax.dynamic_index_in_dim(self.test_puzzles, puzzle_id, keepdims=False),
        )
        return PushWorldPuzzleAll(
            id=puzzle_id,
            a=puzzle[:6].reshape(-1, 2),
            m1=puzzle[6:12].reshape(-1, 2),
            m2=puzzle[12:18].reshape(-1, 2),
            m3=puzzle[18:24].reshape(-1, 2),
            m4=puzzle[24:30].reshape(-1, 2),
            g1=puzzle[30:36].reshape(-1, 2),
            g2=puzzle[36:42].reshape(-1, 2),
            w=puzzle[42:].reshape(-1, 2),
        )


def load_benchmark_from_path(path: str) -> Benchmark:
    puzzle_dict = load_bz2_pickle(path)
    benchmark = Benchmark(
        train_puzzles=puzzle_dict["train"],
        test_puzzles=puzzle_dict["test"],
    )
    return benchmark


def load_all_benchmark_from_path(path: str) -> BenchmarkAll:
    puzzle_dict = load_bz2_pickle(path)
    benchmark = BenchmarkAll(
        train_puzzles=puzzle_dict["train"],
        test_puzzles=puzzle_dict["test"],
    )
    return benchmark


def load_all_benchmark(name: str) -> BenchmarkAll:
    if name not in NAME2HFFILENAME:
        raise RuntimeError(f"Unknown benchmark. Registered: {registered_benchmarks()}")

    os.makedirs(DATA_PATH, exist_ok=True)

    path = os.path.join(DATA_PATH, NAME2HFFILENAME[name])
    if not os.path.exists(path):
        _download_from_hf(HF_REPO_ID, NAME2HFFILENAME[name])

    return load_all_benchmark_from_path(path)


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

    with tqdm(unit="B", unit_scale=True, miniters=1, desc="Progress", disable=True) as t:

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
