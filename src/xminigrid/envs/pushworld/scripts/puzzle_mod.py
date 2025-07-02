# Help me write a script that takes a puzzle file and pads it up to some size, using the logic in upload.py

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyrallis


@dataclass
class Config:
    input: str  # Input .pwp puzzle file path
    output: Optional[str] = None  # Output file path (default: input_padded.pwp)
    size: int = 10  # Target grid size (default: 10)


def pad_puzzle(input_path: str, output_path: str, target_size: int = 10):
    """
    Pad a puzzle file to the target size by adding walls around the border.

    Args:
        input_path: Path to the input .pwp puzzle file
        output_path: Path to save the padded puzzle file
        target_size: Target grid size (default: 10 for 10x10)
    """
    # First pass: determine dimensions
    with open(input_path, "r") as fi:
        lines = [line.strip() for line in fi.readlines() if line.strip()]
        original_height = len(lines)
        original_width = len(lines[0].split()) if lines else 0

    if original_width >= target_size or original_height >= target_size:
        print(
            f"Warning: Original puzzle ({original_width}x{original_height}) is already >= target size ({target_size}x{target_size})"
        )
        print("Copying file without padding...")
        with open(input_path, "r") as src, open(output_path, "w") as dst:
            dst.write(src.read())
        return

    # Calculate padding to center the puzzle
    x_padding_total = target_size - original_width
    y_padding_total = target_size - original_height

    # Center the puzzle by distributing padding
    x_offset = x_padding_total // 2
    y_offset = y_padding_total // 2

    print(f"Original size: {original_width}x{original_height}")
    print(f"Target size: {target_size}x{target_size}")
    print(f"Padding: x_offset={x_offset}, y_offset={y_offset}")

    # Second pass: parse and collect all objects with offsets
    obj_pixels = defaultdict(set)
    for line_idx, line in enumerate(lines):
        y = line_idx + y_offset
        line_elems = line.split()

        for x in range(len(line_elems)):
            elem_id = line_elems[x].upper()  # Keep original case for output
            if elem_id != ".":
                obj_pixels[elem_id].add((x + x_offset, y))

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

    # Add walls to existing walls (if any)
    if "W" in obj_pixels:
        obj_pixels["W"].update(walls)
    else:
        obj_pixels["W"] = walls

    # Create the output grid
    grid = [["." for _ in range(target_size)] for _ in range(target_size)]

    # Fill the grid with objects
    for elem_id, coords in obj_pixels.items():
        for x, y in coords:
            grid[y][x] = elem_id

    # Write the padded puzzle to output file
    with open(output_path, "w") as fo:
        for row in grid:
            fo.write("  ".join(row) + "\n")

    print(f"Padded puzzle saved to: {output_path}")


@pyrallis.wrap()
def main(config: Config):
    input_path = config.input
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' not found")
        return

    # Default output path
    if config.output:
        output_path = config.output
    else:
        input_stem = Path(input_path).stem
        input_dir = Path(input_path).parent
        output_path = input_dir / f"{input_stem}_padded.pwp"

    pad_puzzle(input_path, str(output_path), config.size)


if __name__ == "__main__":
    main()
