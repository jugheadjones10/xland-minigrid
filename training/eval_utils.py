import numpy as np

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
