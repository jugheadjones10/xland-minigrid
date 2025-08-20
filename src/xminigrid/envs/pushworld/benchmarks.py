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


class BenchmarkAllADR(BenchmarkAll):
    train_puzzles_adr: jax.Array
    test_puzzles_adr: jax.Array

    def sample_puzzle(
        self,
        key: jax.Array,
        adr_params: ADRParams,
        type: Literal["train", "test"] = "train",
    ) -> PushWorldPuzzleAll:
        # In sample_puzzle, we randomly pick a puzzle using the key from puzzles that satisfy the ADR params
        pass

    def get_puzzles_subset(self, adr_params: ADRParams, type: Literal["train", "test"] = "train") -> PushWorldPuzzleAll:
        # In get_puzzles_subset, we simply return all puzzles that satisfy the ADR params
        pass


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


def load_all_benchmark_adr(name: str) -> BenchmarkAllADR:
    if name not in NAME2HFFILENAME:
        raise RuntimeError(f"Unknown benchmark. Registered: {registered_benchmarks()}")

    os.makedirs(DATA_PATH, exist_ok=True)

    path = os.path.join(DATA_PATH, NAME2HFFILENAME[name])
    if not os.path.exists(path):
        _download_from_hf(HF_REPO_ID, NAME2HFFILENAME[name])

    puzzle_dict = load_bz2_pickle(path)

    def _reshape_pairs(arr):
        return jnp.asarray(arr, dtype=jnp.int32).reshape(-1, 2)

    def _count_valid(pairs: jax.Array) -> int:
        mask = (pairs[:, 0] != -1) & (pairs[:, 1] != -1)
        return int(jnp.sum(mask))

    def _infer_shape_max(objects: list[jax.Array]) -> int:
        # Shape is maximum block count among agent, movables, goals
        counts = [_count_valid(obj) for obj in objects]
        return int(max(counts)) if len(counts) > 0 else 0

    def _infer_puzzle_size_from_walls(walls_pairs: jax.Array) -> int:
        # Build boolean wall grid from wall coordinates using jnp
        grid = jnp.zeros((LEVEL0_ALL_SIZE, LEVEL0_ALL_SIZE), dtype=bool)
        pairs = jnp.asarray(walls_pairs)
        valid = (pairs[:, 0] != -1) & (pairs[:, 1] != -1)
        x = jnp.clip(pairs[valid, 0], 0, LEVEL0_ALL_SIZE - 1)
        y = jnp.clip(pairs[valid, 1], 0, LEVEL0_ALL_SIZE - 1)
        grid = grid.at[y, x].set(True)

        # Bounding box over non-walls
        non_wall = jnp.logical_not(grid)
        has_non_wall = jnp.any(non_wall)
        if not bool(has_non_wall):
            return 0
        ys, xs = jnp.where(non_wall)
        min_x = int(jnp.min(xs))
        max_x = int(jnp.max(xs))
        min_y = int(jnp.min(ys))
        max_y = int(jnp.max(ys))

        width = max_x - min_x + 1
        height = max_y - min_y + 1
        return int(max(width, height))

    def _count_internal_walls(walls_pairs: jax.Array) -> int:
        # Build boolean wall grid
        grid = jnp.zeros((LEVEL0_ALL_SIZE, LEVEL0_ALL_SIZE), dtype=bool)
        pairs = jnp.asarray(walls_pairs)
        valid = (pairs[:, 0] != -1) & (pairs[:, 1] != -1)
        x = jnp.clip(pairs[valid, 0], 0, LEVEL0_ALL_SIZE - 1)
        y = jnp.clip(pairs[valid, 1], 0, LEVEL0_ALL_SIZE - 1)
        grid = grid.at[y, x].set(True)

        # Bounding box over non-walls
        non_wall = jnp.logical_not(grid)
        ys, xs = jnp.where(non_wall)
        min_x = int(jnp.min(xs))
        max_x = int(jnp.max(xs))
        min_y = int(jnp.min(ys))
        max_y = int(jnp.max(ys))

        internal = grid[min_y : max_y + 1, min_x : max_x + 1]
        return int(jnp.sum(internal))

    def _build_adr_rows(puzzles: jax.Array) -> jax.Array:
        rows = []
        for puzzle in puzzles:
            # Slice like BenchmarkAll.get_puzzle
            a = _reshape_pairs(puzzle[:6])
            m1 = _reshape_pairs(puzzle[6:12])
            m2 = _reshape_pairs(puzzle[12:18])
            m3 = _reshape_pairs(puzzle[18:24])
            m4 = _reshape_pairs(puzzle[24:30])
            g1 = _reshape_pairs(puzzle[30:36])
            g2 = _reshape_pairs(puzzle[36:42])
            w = _reshape_pairs(puzzle[42:])

            # ADR attributes
            puzzle_size = _infer_puzzle_size_from_walls(w)
            num_walls = _count_internal_walls(w)
            # Count only distractor movables (exclude goal movables based on g1/g2 presence)
            m1_exists = _count_valid(m1) > 0
            m2_exists = _count_valid(m2) > 0
            m3_exists = _count_valid(m3) > 0
            m4_exists = _count_valid(m4) > 0
            g1_exists = _count_valid(g1) > 0
            g2_exists = _count_valid(g2) > 0

            if g1_exists and g2_exists:
                # m1 and m2 are goal movables; m3, m4 are distractors
                num_movables = int(m3_exists) + int(m4_exists)
            elif g1_exists and not g2_exists:
                # m1 is goal movable; m2, m3, m4 are distractors
                num_movables = int(m2_exists) + int(m3_exists) + int(m4_exists)

            shape = _infer_shape_max([a, m1, m2, m3, m4, g1, g2])
            num_goals = int((_count_valid(g1) > 0) + (_count_valid(g2) > 0))

            rows.append([puzzle_size, num_walls, num_movables, shape, num_goals])

        return jnp.asarray(rows, dtype=jnp.int32)

    train_puzzles_adr = (
        _build_adr_rows(puzzle_dict["train"]) if len(puzzle_dict["train"]) > 0 else jnp.zeros((0, 5), dtype=jnp.int32)
    )
    test_puzzles_adr = (
        _build_adr_rows(puzzle_dict["test"]) if len(puzzle_dict["test"]) > 0 else jnp.zeros((0, 5), dtype=jnp.int32)
    )

    benchmark = BenchmarkAllADR(
        train_puzzles=puzzle_dict["train"],
        test_puzzles=puzzle_dict["test"],
        train_puzzles_adr=train_puzzles_adr,
        test_puzzles_adr=test_puzzles_adr,
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
