import os
from typing import Optional

import jax
import numpy as np
import orbax.checkpoint
import wandb
from orbax.checkpoint import utils as orbax_utils

from xminigrid.envs.pushworld.constants import Tiles


def hex_to_rgb(hex_string: str):
    """Converts a standard 6-digit hex color into a tuple of decimal
    (red, green, blue) values."""
    return tuple(int(hex_string[i : i + 2], 16) for i in (0, 2, 4))


symbol_to_rgb = {
    0: hex_to_rgb("FFFFFF"),  # empty → white
    1: hex_to_rgb("00DC00"),  # agent → "00DC00"
    2: hex_to_rgb("469BFF"),  # movable → "469BFF"
    3: hex_to_rgb("DC0000"),  # movable_goal → "DC0000"
    4: hex_to_rgb("0A0A0A"),  # wall → "0A0A0A"
}


def text_to_rgb(goal_pos, grid):
    """grid: 2-D array of str, shape (H, W)"""
    h, w = grid.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for sym, rgb in symbol_to_rgb.items():
        mask = grid == sym
        img[mask] = rgb

    if grid[goal_pos[1], goal_pos[0]] == Tiles.EMPTY:
        img[goal_pos[1], goal_pos[0]] = hex_to_rgb("FF7F7F")  # light red

    # upscale (optional) so each tile is, say, 16×16 pixels
    img = np.kron(img, np.ones((64, 64, 1), dtype=np.uint8))
    return img


def text_to_rgb_all(observation: jax.Array):
    # I want you to render the observation into a grid
    # Observation is a jax.Array, shape (H, W, 8),
    # Where 8 is the number of channels.
    # Each channel represents a different object, which should have its own color.
    # This is the order of the channels:
    # channels.append(create_channel(state.a))  # agent
    # channels.append(create_channel(state.m1))  # movable 1
    # channels.append(create_channel(state.m2))  # movable 2
    # channels.append(create_channel(state.m3))  # movable 3
    # channels.append(create_channel(state.m4))  # movable 4
    # channels.append(create_channel(puzzle.g1))  # goal 1
    # channels.append(create_channel(puzzle.g2))  # goal 2
    # channels.append(create_channel(puzzle.w))  # walls

    # Movables that have associated goals should be given the "movable_goal" color,
    # movables that do not should just be given the "movable" color.

    # Convert to numpy for easier processing
    obs_np = np.array(observation)
    h, w = obs_np.shape[:2]

    # Create RGB image initialized to white (empty spaces)
    rgb_img = np.zeros((h, w, 3), dtype=np.uint8)
    rgb_img.fill(255)  # white background

    # Channel indices
    AGENT_CH = 0
    M1_CH, M2_CH, M3_CH, M4_CH = 1, 2, 3, 4
    G1_CH, G2_CH = 5, 6
    WALL_CH = 7

    # Extract individual channels
    agent = obs_np[:, :, AGENT_CH]
    m1 = obs_np[:, :, M1_CH]
    m2 = obs_np[:, :, M2_CH]
    m3 = obs_np[:, :, M3_CH]
    m4 = obs_np[:, :, M4_CH]
    g1 = obs_np[:, :, G1_CH]
    g2 = obs_np[:, :, G2_CH]
    walls = obs_np[:, :, WALL_CH]

    # Render walls first (bottom layer)
    wall_mask = walls > 0
    rgb_img[wall_mask] = symbol_to_rgb[4]  # black

    # Render goals (light red for empty goals)
    g1_mask = g1 > 0
    g2_mask = g2 > 0
    rgb_img[g1_mask] = hex_to_rgb("FF7F7F")  # light red
    rgb_img[g2_mask] = hex_to_rgb("FF7F7F")  # light red

    # Render movables with appropriate colors
    # Goal-movable pairing is dynamic based on which goals exist:
    # - If only g1 exists: m1 is the goal movable, m2/m3/m4 are regular movables
    # - If both g1 and g2 exist: m1 and m2 are goal movables, m3/m4 are regular movables

    # Check which goals exist
    g1_exists = np.any(g1_mask)
    g2_exists = np.any(g2_mask)

    # m1: movable_goal color if on g1. g1 is guaranteed to always exist.
    m1_mask = m1 > 0
    rgb_img[m1_mask] = symbol_to_rgb[3]  # movable_goal (red)

    # m2: movable_goal color if on g2 (when g2 exists), otherwise movable color
    m2_mask = m2 > 0
    if g2_exists:
        rgb_img[m2_mask] = symbol_to_rgb[3]  # movable_goal (red)
    else:
        rgb_img[m2_mask] = symbol_to_rgb[2]  # movable (blue)

    # m3 and m4: always regular movable color (no associated goals)
    m3_mask = m3 > 0
    m4_mask = m4 > 0
    rgb_img[m3_mask] = symbol_to_rgb[2]  # movable (blue)
    rgb_img[m4_mask] = symbol_to_rgb[2]  # movable (blue)

    # Render agent on top
    agent_mask = agent > 0
    rgb_img[agent_mask] = symbol_to_rgb[1]  # agent (green)

    # Upscale for better visibility (64x64 pixels per tile)
    upscaled_img = np.kron(rgb_img, np.ones((64, 64, 1), dtype=np.uint8))

    return upscaled_img


def fetch_model_from_wandb(
    project: str,
    run_id: str,
    artifact_name: Optional[str] = None,
    download_path: str = "./downloaded_checkpoints",
    entity: Optional[str] = None,
):
    """
    Fetch model parameters from a WandB artifact.

    Args:
        project: WandB project name
        run_id: WandB run ID
        artifact_name: Specific artifact name (if None, uses f"model-checkpoint-{run_id}")
        download_path: Local path to download the artifact
        entity: WandB entity/username (if None, uses default)

    Returns:
        Model parameters from the checkpoint
    """

    # Initialize wandb in offline mode to avoid creating a new run
    if entity:
        wandb.init(project=project, entity=entity, mode="offline")
    else:
        wandb.init(project=project, mode="offline")

    try:
        # Construct artifact name if not provided
        if artifact_name is None:
            artifact_name = f"model-checkpoint-{run_id}:latest"
        elif ":latest" not in artifact_name and ":v" not in artifact_name:
            artifact_name = f"{artifact_name}:latest"

        print(f"Fetching artifact: {artifact_name}")

        # Download the artifact
        artifact = wandb.use_artifact(artifact_name)
        artifact_dir = artifact.download(root=download_path)

        print(f"Downloaded to: {artifact_dir}")

        # Load the checkpoint using orbax
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        checkpoint = orbax_checkpointer.restore(artifact_dir)

        print("Successfully loaded checkpoint")

        return checkpoint["params"]

    except Exception as e:
        print(f"Error fetching model from WandB: {e}")
        raise
    finally:
        wandb.finish()
