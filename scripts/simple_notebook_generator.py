#!/usr/bin/env python3
"""
Simple Notebook Generator

Just specify source files and training config. Examples:

# Single task training:
python simple_notebook_generator.py \
    --train_file=train_single_task_pushworld.py \
    --eval_file=eval_single_pushworld.py \
    --output_name=single_task

# Meta task training:  
python simple_notebook_generator.py \
    --train_file=train_meta_task_pushworld.py \
    --eval_file=eval_meta_pushworld.py \
    --output_name=meta_task \
    --train_config='config = TrainConfig(benchmark_id="level0_transformed_base", total_timesteps=1000_000_000, num_envs=8192, num_steps_per_env=500, num_steps_per_update=500, train_test_same=False, num_train=2000, num_test=200)'

# Using "All" versions:
python simple_notebook_generator.py \
    --train_file=train_single_task_pushworld_all.py \
    --utils_file=utils_pushworld_all.py \
    --eval_file=eval_single_pushworld.py \
    --output_name=single_task_all
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import libcst as cst
import nbformat as nbf
import pyrallis

TRAINING_DIR = Path(__file__).parent.parent / "training"
EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"


# Example preset configurations
PRESETS = {
    "single": {
        "train_file": "train_single_task_pushworld.py",
        "eval_file": "eval_single_pushworld.py",
        "output_name": "pushworld_single_task",
        "title": "Single-task PushWorld Training",
        "train_config": """config = TrainConfig(
    benchmark_id="level0_transformed_base", 
    total_timesteps=200_000_000, 
    num_envs=8192, 
    num_steps=100
)""",
    },
    "meta": {
        "train_file": "train_meta_task_pushworld.py",
        "eval_file": "eval_meta_pushworld.py",
        "output_name": "pushworld_meta_task",
        "title": "Meta-task PushWorld Training",
        "train_config": """config = TrainConfig(
    benchmark_id="level0_transformed_base",
    total_timesteps=1000_000_000,
    num_envs=8192,
    num_steps_per_env=500,
    num_steps_per_update=500,
    train_test_same=False,
    num_train=2000,
    num_test=200,
)""",
    },
    "single_all": {
        "train_file": "train_single_task_pushworld_all.py",
        "utils_file": "utils_pushworld_all.py",
        "eval_file": "eval_single_pushworld.py",
        "output_name": "pushworld_single_task_all",
        "title": "Single-task PushWorld Training (All Environment)",
        "train_config": """config = TrainConfig(
    benchmark_id="level0_transformed_base",
    total_timesteps=1000_000_000,
    num_envs=8192,
    num_steps=100
)""",
    },
    "meta_all": {
        "train_file": "train_meta_task_pushworld_all.py",
        "utils_file": "utils_pushworld_all.py",
        # TODO: add eval_meta_pushworld_all.py
        "eval_file": "eval_meta_pushworld.py",
        "output_name": "pushworld_meta_task_all",
        "title": "Meta-task PushWorld Training (All Environment)",
        "train_config": """config = TrainConfig(
    benchmark_id="level0_transformed_base",
    total_timesteps=1000_000_000,
    num_envs=8192,
    num_steps_per_env=500,
    num_steps_per_update=500,
    train_test_same=False,
    num_train=2000,
    num_test=200,
)""",
    },
}


@dataclass
class Config:
    # Preset configuration (if specified, overrides individual settings)
    preset: Optional[str] = None

    # Core source files
    nn_file: str = "nn_pushworld.py"
    utils_file: str = "utils_pushworld.py"
    train_file: str = "train_single_task_pushworld.py"
    eval_utils_file: str = "eval_utils.py"
    eval_file: str = "eval_single_pushworld.py"

    # Training configuration code (as string)
    train_config: str = """config = TrainConfig(
    benchmark_id="level0_transformed_base", 
    total_timesteps=200_000_000, 
    num_envs=8192, 
    num_steps=100
)"""

    # Output filename (without extension)
    output_name: str = "pushworld_training"

    # Optional title for the notebook
    title: Optional[str] = None

    def __post_init__(self):
        """Apply preset configuration if specified."""
        if self.preset and self.preset in PRESETS:
            preset_config = PRESETS[self.preset]
            for key, value in preset_config.items():
                if not hasattr(self, key) or getattr(self, key) == getattr(Config(), key, None):
                    # Only override if using default value
                    setattr(self, key, value)

    @property
    def notebook_title(self) -> str:
        """Get the title for the notebook."""
        if self.title:
            return self.title
        # Generate title from train_file
        name_part = self.train_file.replace("train_", "").replace(".py", "").replace("_", " ").title()
        return f"{name_part} Training"


# Preprocessing functions
def remove_imports(code):
    """Remove import statements from code."""
    tree = cst.parse_module(code)

    class ImportRemover(cst.CSTTransformer):
        def leave_SimpleStatementLine(self, original_node, updated_node):
            if isinstance(updated_node.body[0], (cst.Import, cst.ImportFrom)):
                return cst.RemovalSentinel.REMOVE
            return updated_node

    new_tree = tree.visit(ImportRemover())
    return new_tree.code


def remove_main_block(code):
    """Remove if __name__ == '__main__': blocks."""
    tree = cst.parse_module(code)

    class MainBlockRemover(cst.CSTTransformer):
        def leave_If(self, original_node, updated_node):
            if isinstance(updated_node.test, cst.Comparison):
                comparison = updated_node.test
                if isinstance(comparison.left, cst.Name) and comparison.left.value == "__name__":
                    return cst.RemovalSentinel.REMOVE
            return updated_node

    new_tree = tree.visit(MainBlockRemover())
    return new_tree.code


def extract_functions_except(code, exclude_functions=None):
    """Extract all functions except specified ones."""
    if exclude_functions is None:
        exclude_functions = []

    tree = cst.parse_module(code)

    class FunctionExtractor(cst.CSTTransformer):
        def __init__(self):
            self.functions_to_remove = set(exclude_functions)

        def leave_FunctionDef(self, original_node, updated_node):
            # Remove excluded functions (this handles the function + its decorators)
            if updated_node.name.value in self.functions_to_remove:
                return cst.RemovalSentinel.REMOVE
            return updated_node

        def leave_If(self, original_node, updated_node):
            # Remove if __name__ == "__main__": blocks
            if isinstance(updated_node.test, cst.Comparison):
                comparison = updated_node.test
                if isinstance(comparison.left, cst.Name) and comparison.left.value == "__name__":
                    return cst.RemovalSentinel.REMOVE
            return updated_node

    new_tree = tree.visit(FunctionExtractor())
    return new_tree.code


def extract_specific_functions(code, include_functions=None):
    """Extract only specified functions."""
    if include_functions is None:
        include_functions = []

    tree = cst.parse_module(code)

    class SpecificFunctionExtractor(cst.CSTTransformer):
        def leave_SimpleStatementLine(self, original_node, updated_node):
            # Remove import statements
            if isinstance(updated_node.body[0], (cst.Import, cst.ImportFrom)):
                return cst.RemovalSentinel.REMOVE
            return updated_node

        def leave_FunctionDef(self, original_node, updated_node):
            # Keep only included functions
            if updated_node.name.value not in include_functions:
                return cst.RemovalSentinel.REMOVE
            return updated_node

        def leave_ClassDef(self, original_node, updated_node):
            # Remove all classes
            return cst.RemovalSentinel.REMOVE

        def leave_If(self, original_node, updated_node):
            # Remove if __name__ == "__main__": blocks
            if isinstance(updated_node.test, cst.Comparison):
                comparison = updated_node.test
                if isinstance(comparison.left, cst.Name) and comparison.left.value == "__name__":
                    return cst.RemovalSentinel.REMOVE
            return updated_node

    new_tree = tree.visit(SpecificFunctionExtractor())
    return new_tree.code


def remove_top_level_comments(code):
    """Remove top-level comments from code."""
    tree = cst.parse_module(code)

    class CommentRemover(cst.CSTTransformer):
        def leave_SimpleStatementLine(self, original_node, updated_node):
            # Remove leading comments from statements
            if updated_node.leading_lines:
                # Filter out comment lines, keep empty lines
                new_leading_lines = []
                for line in updated_node.leading_lines:
                    if isinstance(line, cst.EmptyLine) and line.comment is None:
                        new_leading_lines.append(line)
                    # Skip lines that have comments
                updated_node = updated_node.with_changes(leading_lines=new_leading_lines)
            return updated_node

        def leave_Module(self, original_node, updated_node):
            # Remove header comments from the module
            if updated_node.header:
                # Filter out comment lines from the header
                new_header = []
                for line in updated_node.header:
                    if isinstance(line, cst.EmptyLine) and line.comment is None:
                        new_header.append(line)
                    # Skip lines that have comments
                updated_node = updated_node.with_changes(header=new_header)
            return updated_node

    new_tree = tree.visit(CommentRemover())
    return new_tree.code


def extract_execution_code(code):
    """Extract the contents of the main function."""
    tree = cst.parse_module(code)

    class MainFunctionExtractor(cst.CSTVisitor):
        def __init__(self):
            self.main_function_body = None

        def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
            if node.name.value == "main":
                # Create a temporary module with just the body to get the code
                temp_module = tree.with_changes(body=node.body.body)
                self.main_function_body = temp_module.code.strip()

    extractor = MainFunctionExtractor()
    tree.visit(extractor)

    return extractor.main_function_body


def extract_training_main_functions(code):
    """Extract main training functions (everything except processing and main)."""
    code = remove_top_level_comments(code)
    code = remove_imports(code)
    code = remove_main_block(code)
    code = extract_functions_except(code, exclude_functions=["processing", "main"])
    return code


def extract_processing_function(code):
    """Extract only the processing function."""
    code = remove_top_level_comments(code)
    return extract_specific_functions(code, include_functions=["processing"])


def extract_init_code(code):
    code = remove_top_level_comments(code)
    code = extract_execution_code(code)
    return code


# Registry of available processors
PROCESSORS = {
    "remove_imports": remove_imports,
    "remove_main_block": remove_main_block,
    "extract_training_main_functions": extract_training_main_functions,
    "extract_processing_function": extract_processing_function,
    "extract_init_code": extract_init_code,
}


def apply_processors(code, processor_names):
    """Apply a list of processors to code."""
    for processor_name in processor_names:
        if processor_name in PROCESSORS:
            code = PROCESSORS[processor_name](code)
        else:
            print(f"Warning: Unknown processor '{processor_name}'")
    return code


def generate_notebook(cells, source_dir, output_path):
    """
    Generate a notebook from a simple cell specification.

    cells: list of dicts with either:
        - {"type": "markdown", "content": "markdown text"}
        - {"type": "code", "files": ["path/to/file.py"]} (relative to training_dir if provided)
        - {"type": "code", "content": "direct code"}
    training_dir: base directory for resolving relative file paths
    """
    nb = nbf.v4.new_notebook()

    for cell_spec in cells:
        if cell_spec["type"] == "markdown":
            nb.cells.append(nbf.v4.new_markdown_cell(cell_spec["content"]))

        elif cell_spec["type"] == "code":
            if "files" in cell_spec:
                # Read code from files
                code = ""
                for file_path in cell_spec["files"]:
                    full_path = source_dir / file_path

                    with open(full_path, "r") as f:
                        file_code = f.read()

                    # Apply preprocessing
                    file_code = apply_processors(
                        file_code,
                        cell_spec.get("processors", []),
                    )

                    code += file_code + "\n\n"
                code = code.strip()
            else:
                # Direct code content
                code = cell_spec["content"]

            nb.cells.append(nbf.v4.new_code_cell(code))

    # Save notebook
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        nbf.write(nb, f)

    print(f"Generated notebook: {output_path}")
    return output_path


# Example usage for PushWorld
def generate_pushworld_notebook(source_dir, output_dir, config: Config):
    cells = [
        {
            "type": "markdown",
            "content": f"# {config.notebook_title}",
        },
        {
            "type": "code",
            "content": '# Install if needed\n# !pip install "xminigrid[baselines] @ git+https://github.com/jugheadjones10/xland-minigrid.git"',
        },
        {
            "type": "code",
            "content": """import os
import shutil
import time #noqa
import os #noqa
import math # noqa
from typing import TypedDict, Optional, Literal #noqa
import numpy as np #noqa
import importlib #noqa
import os #noqa

import jax #noqa
import jax.numpy as jnp #noqa
import jax.tree_util as jtu #noqa
import flax #noqa
import flax.linen as nn #noqa
from flax.training import orbax_utils #noqa
import distrax #noqa
import orbax #noqa
import optax #noqa
import imageio #noqa
import wandb #noqa
import matplotlib.pyplot as plt #noqa

from flax import struct #noqa
from flax.typing import Dtype #noqa
from flax.linen.dtypes import promote_dtype #noqa
from flax.linen.initializers import glorot_normal, orthogonal, zeros_init #noqa
from flax.training.train_state import TrainState #noqa
from flax.jax_utils import replicate, unreplicate #noqa
from dataclasses import asdict, dataclass #noqa
from functools import partial #noqa

import xminigrid.envs.pushworld as pushworld
from xminigrid.envs.pushworld.benchmarks import Benchmark
from xminigrid.envs.pushworld.constants import Tiles, NUM_TILES, SUCCESS_REWARD
from xminigrid.envs.pushworld.environment import Environment, EnvParams, EnvParamsT
from xminigrid.envs.pushworld.envs.single_task_pushworld import SingleTaskPushWorldEnvironment, SingleTaskPushWorldEnvParams
from xminigrid.envs.pushworld.envs.meta_task_pushworld import MetaTaskPushWorldEnvironment
from xminigrid.envs.pushworld.scripts.upload import encode_puzzle
from xminigrid.envs.pushworld.wrappers import GoalObservationWrapper, GymAutoResetWrapper, Wrapper
from xminigrid.envs.pushworld.types import State, TimeStep, StepType, EnvCarry, PushWorldPuzzle
from xminigrid.envs.pushworld.grid import get_obs_from_puzzle
from IPython.display import Video, HTML, display""",
        },
        {"type": "markdown", "content": "## Networks"},
        {"type": "code", "files": [config.nn_file], "processors": ["remove_imports"]},
        {"type": "markdown", "content": "## Utils"},
        {"type": "code", "files": [config.utils_file], "processors": ["remove_imports"]},
        {"type": "markdown", "content": "## Training"},
        {
            "type": "code",
            "files": [config.train_file],
            "processors": ["extract_training_main_functions"],
        },
        {"type": "markdown", "content": "## Processing"},
        {"type": "code", "files": [config.train_file], "processors": ["extract_processing_function"]},
        {"type": "markdown", "content": "## Evaluation"},
        {"type": "code", "files": [config.eval_utils_file], "processors": ["remove_imports"]},
        {
            "type": "code",
            "files": [config.eval_file],
            "processors": ["remove_imports"],
        },
        {"type": "markdown", "content": "## Run Training"},
        {
            "type": "code",
            "content": config.train_config,
        },
        {"type": "code", "files": [config.train_file], "processors": ["extract_init_code"]},
    ]

    return generate_notebook(cells, source_dir, output_dir / f"{config.output_name}_base.ipynb")


@pyrallis.wrap()
def main(config: Config):
    generate_pushworld_notebook(TRAINING_DIR, EXPERIMENTS_DIR, config)


if __name__ == "__main__":
    main()
