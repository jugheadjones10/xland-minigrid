from __future__ import annotations

import abc
import bz2
import os
import pickle
import urllib.request
from functools import partial
from typing import Generic, Literal

import jax
import jax.numpy as jnp
from flax import struct
from tqdm.auto import tqdm

from .benchmarks import NAME2HFFILENAME, BenchmarkAll, _download_from_hf, load_bz2_pickle
from .constants import LEVEL0_ALL_SIZE
from .types import ADRParams, PushWorldPuzzle, PushWorldPuzzleAll, PuzzleT


class BenchmarkAllADR(BenchmarkAll):
    train_puzzles_adr: jax.Array
    test_puzzles_adr: jax.Array

    @partial(
        jax.jit,
        static_argnums=(3,),
    )
    def sample_puzzle(
        self,
        key: jax.Array,
        adr_params: ADRParams,
        type: Literal["train", "test"] = "train",
    ) -> PushWorldPuzzleAll:
        # In sample_puzzle, we randomly pick a puzzle using the key from puzzles that satisfy the ADR params
        # Filter for the indexes that match the adr_params, then return the puzzles

        if type == "train":
            adr = self.train_puzzles_adr
        else:
            adr = self.test_puzzles_adr

        valid_mask = (
            (adr[:, 0] >= adr_params.puzzle_size[0])
            & (adr[:, 0] <= adr_params.puzzle_size[1])
            & (adr[:, 1] >= adr_params.num_walls[0])
            & (adr[:, 1] <= adr_params.num_walls[1])
            & (adr[:, 2] >= adr_params.num_movables[0])
            & (adr[:, 2] <= adr_params.num_movables[1])
            & (adr[:, 3] >= adr_params.shape[0])
            & (adr[:, 3] <= adr_params.shape[1])
            & (adr[:, 4] >= adr_params.num_goals[0])
            & (adr[:, 4] <= adr_params.num_goals[1])
        )

        return self.get_puzzle(jax.random.choice(key, adr.shape[0], shape=(), p=valid_mask), type)

    @partial(
        jax.jit,
        static_argnums=(2,),
    )
    def get_puzzles_subset_mask(self, adr_params: ADRParams, type: Literal["train", "test"] = "train") -> jax.Array:
        if type == "train":
            adr = self.train_puzzles_adr
        else:
            adr = self.test_puzzles_adr

        valid_mask = (
            (adr[:, 0] >= adr_params.puzzle_size[0])
            & (adr[:, 0] <= adr_params.puzzle_size[1])
            & (adr[:, 1] >= adr_params.num_walls[0])
            & (adr[:, 1] <= adr_params.num_walls[1])
            & (adr[:, 2] >= adr_params.num_movables[0])
            & (adr[:, 2] <= adr_params.num_movables[1])
            & (adr[:, 3] >= adr_params.shape[0])
            & (adr[:, 3] <= adr_params.shape[1])
            & (adr[:, 4] >= adr_params.num_goals[0])
            & (adr[:, 4] <= adr_params.num_goals[1])
        )

        return valid_mask


# Vectorized ADR computation with JAX vmap/jit (no Python loops)
HF_REPO_ID = os.environ.get("PUSHWORLD_HF_REPO_ID", "feynmaniac/pushworld")
DATA_PATH = os.environ.get("PUSHWORLD_DATA", os.path.expanduser("~/.pushworld"))

NAME2HFFILENAME = {
    "level0_transformed_all": "pushworld_level0_transformed_all.pkl",
    "level0_transformed_base": "pushworld_level0_transformed_base.pkl",
    "level0_mini": "pushworld_level0_mini.pkl",
}


# Helper to slice flat puzzles into (K,2) pairs per object
def _slice_pairs(flat: jnp.ndarray, start: int, end: int) -> jnp.ndarray:
    return flat[start:end].reshape(-1, 2)


def _count_valid_pairs(pairs: jnp.ndarray) -> jnp.ndarray:
    valid = (pairs[:, 0] != -1) & (pairs[:, 1] != -1)
    return jnp.sum(valid, dtype=jnp.int32)


def _compute_adr_one(flat: jnp.ndarray) -> jnp.ndarray:
    # Slice
    a = _slice_pairs(flat, 0, 6)
    m1 = _slice_pairs(flat, 6, 12)
    m2 = _slice_pairs(flat, 12, 18)
    m3 = _slice_pairs(flat, 18, 24)
    m4 = _slice_pairs(flat, 24, 30)
    g1 = _slice_pairs(flat, 30, 36)
    g2 = _slice_pairs(flat, 36, 42)
    w = _slice_pairs(flat, 42, flat.shape[0])

    # Walls bbox from column/row completeness (no grid construction)
    valid_w = (w[:, 0] != -1) & (w[:, 1] != -1)
    x = jnp.clip(w[:, 0], 0, LEVEL0_ALL_SIZE - 1)
    y = jnp.clip(w[:, 1], 0, LEVEL0_ALL_SIZE - 1)

    col_counts = jnp.bincount(x, weights=valid_w.astype(jnp.int32), length=LEVEL0_ALL_SIZE)
    row_counts = jnp.bincount(y, weights=valid_w.astype(jnp.int32), length=LEVEL0_ALL_SIZE)

    # Columns/rows that are NOT fully walls (i.e., have at least one non-wall)
    non_wall_cols = col_counts < LEVEL0_ALL_SIZE  # shape (10,)
    non_wall_rows = row_counts < LEVEL0_ALL_SIZE  # shape (10,)

    idxs = jnp.arange(LEVEL0_ALL_SIZE, dtype=jnp.int32)

    # Static-shape min/max by masking indices with sentinels
    min_x = jnp.min(jnp.where(non_wall_cols, idxs, jnp.full_like(idxs, LEVEL0_ALL_SIZE)))
    max_x = jnp.max(jnp.where(non_wall_cols, idxs, jnp.full_like(idxs, -1)))
    min_y = jnp.min(jnp.where(non_wall_rows, idxs, jnp.full_like(idxs, LEVEL0_ALL_SIZE)))
    max_y = jnp.max(jnp.where(non_wall_rows, idxs, jnp.full_like(idxs, -1)))

    # Width/height; clamp to >= 0 in degenerate cases
    width = jnp.maximum(0, max_x - min_x + 1)
    height = jnp.maximum(0, max_y - min_y + 1)
    puzzle_size = jnp.maximum(width, height).astype(jnp.int32)

    # Internal walls (inside bbox, including border)
    in_bbox = valid_w & (x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y)
    num_walls = jnp.sum(in_bbox, dtype=jnp.int32)

    a_valid_pairs = _count_valid_pairs(a)
    m1_valid_pairs = _count_valid_pairs(m1)
    m2_valid_pairs = _count_valid_pairs(m2)
    m3_valid_pairs = _count_valid_pairs(m3)
    m4_valid_pairs = _count_valid_pairs(m4)
    g1_valid_pairs = _count_valid_pairs(g1)
    g2_valid_pairs = _count_valid_pairs(g2)

    # Movables/goals existence
    m1_exists = (m1_valid_pairs > 0).astype(jnp.int32)
    m2_exists = (m2_valid_pairs > 0).astype(jnp.int32)
    m3_exists = (m3_valid_pairs > 0).astype(jnp.int32)
    m4_exists = (m4_valid_pairs > 0).astype(jnp.int32)
    g1_exists = (g1_valid_pairs > 0).astype(jnp.int32)
    g2_exists = (g2_valid_pairs > 0).astype(jnp.int32)

    num_movables = (m1_exists + m2_exists + m3_exists + m4_exists) - g1_exists * m1_exists - g2_exists * m2_exists

    # Shape = max blocks among a, m1..m4, g1, g2
    shape = jnp.max(
        jnp.array(
            [
                a_valid_pairs,
                m1_valid_pairs,
                m2_valid_pairs,
                m3_valid_pairs,
                m4_valid_pairs,
                g1_valid_pairs,
                g2_valid_pairs,
            ],
            dtype=jnp.int32,
        )
    )

    num_goals = g1_exists + g2_exists

    return jnp.array([puzzle_size, num_walls, num_movables, shape, num_goals], dtype=jnp.int32)


_compute_adr_batch = jax.jit(jax.vmap(_compute_adr_one, in_axes=(0,)))


def load_all_benchmark_adr(name: str) -> BenchmarkAllADR:
    if name not in NAME2HFFILENAME:
        raise RuntimeError("Unknown benchmark")

    os.makedirs(DATA_PATH, exist_ok=True)
    path = os.path.join(DATA_PATH, NAME2HFFILENAME[name])
    if not os.path.exists(path):
        _download_from_hf(HF_REPO_ID, NAME2HFFILENAME[name])

    puzzle_dict = load_bz2_pickle(path)
    train = jnp.asarray(puzzle_dict["train"])
    test = jnp.asarray(puzzle_dict["test"])

    train_adr = _compute_adr_batch(train) if train.shape[0] > 0 else jnp.zeros((0, 5), dtype=jnp.int32)
    test_adr = _compute_adr_batch(test) if test.shape[0] > 0 else jnp.zeros((0, 5), dtype=jnp.int32)

    return BenchmarkAllADR(
        train_puzzles=train,
        test_puzzles=test,
        train_puzzles_adr=train_adr,
        test_puzzles_adr=test_adr,
    )
