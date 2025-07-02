# Help me test the take_action_all function using pytest

import os
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple
from unittest.mock import Mock

import jax
import jax.numpy as jnp
import pytest

from .actions_all import ID_TO_CHANNEL, take_action_all
from .constants import LEVEL0_ALL_SIZE
from .observation import create_channel
from .scripts.upload import encode_puzzle
from .types import EnvCarry, PushWorldPuzzleAll, StateAll


def parse_puzzle_file(puzzle_path: str) -> tuple[Any, Any, jax.Array]:
    """
    Parse a .pwp puzzle file and return puzzle, state, and observation.

    Returns:
        tuple: (PuzzleAll object, StateAll object, observation array)
    """
    puzzle = encode_puzzle(puzzle_path)
    puzzle = PuzzleAll(
        id=jnp.array(0),
        a=puzzle[:6].reshape(-1, 2),
        m1=puzzle[6:12].reshape(-1, 2),
        m2=puzzle[12:18].reshape(-1, 2),
        m3=puzzle[18:24].reshape(-1, 2),
        m4=puzzle[24:30].reshape(-1, 2),
        g1=puzzle[30:36].reshape(-1, 2),
        g2=puzzle[36:42].reshape(-1, 2),
        w=puzzle[42:].reshape(-1, 2),
    )

    # Create StateAll object using actual PyTree class
    state = StateAll(
        key=jnp.array([0]),
        step_num=jnp.array(0),
        a=puzzle.a,
        m1=puzzle.m1,
        m2=puzzle.m2,
        m3=puzzle.m3,
        m4=puzzle.m4,
        carry=EnvCarry(),
    )

    # Create observation (8-channel grid)
    observation = jnp.stack(
        [
            create_channel(puzzle.a),  # agent
            create_channel(puzzle.m1),  # movable 1
            create_channel(puzzle.m2),  # movable 2
            create_channel(puzzle.m3),  # movable 3
            create_channel(puzzle.m4),  # movable 4
            create_channel(puzzle.g1),  # goal 1
            create_channel(puzzle.g2),  # goal 2
            create_channel(puzzle.w),  # walls
        ],
        axis=-1,
    )

    return puzzle, state, observation


class TestCase(NamedTuple):
    """Test case definition with initial state, action, and expected final state."""

    name: str
    initial_puzzle_path: str
    final_puzzle_path: str
    action: int
    description: str = ""


def compare_states(actual_state: Any, expected_puzzle: Any, test_name: str) -> bool:
    """
    Compare actual state after action with expected puzzle state.

    Args:
        actual_state: The state after applying action
        expected_puzzle: The expected puzzle state
        test_name: Name of test for error reporting

    Returns:
        bool: True if states match
    """
    # Compare each object type
    object_types = ["a", "m1", "m2", "m3", "m4"]

    for obj_type in object_types:
        actual_coords = getattr(actual_state, obj_type, None)
        expected_coords = getattr(expected_puzzle, obj_type, None)

        if actual_coords is None or expected_coords is None:
            continue

        if not jnp.array_equal(actual_coords, expected_coords):
            print(f"❌ {test_name}: {obj_type} mismatch")
            print(f"   Expected: {expected_coords.reshape(-1, 2)}")
            print(f"   Actual:   {actual_coords.reshape(-1, 2)}")
            return False

    print(f"✅ {test_name}: States match!")
    return True


# Create a simple mock PuzzleAll class for testing
class PuzzleAll:
    def __init__(self, id, a, m1, m2, m3, m4, g1, g2, w):
        self.id = id
        self.a = a
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.m4 = m4
        self.g1 = g1
        self.g2 = g2
        self.w = w


# Different test cases:
# Agent moving left, right, up, down normally with no wall.
# Agent moving into puzzle boundary when there's no wall.
# Agent moving into wall and getting stopped.
# Agent pushing a single movable object up, down, left, right.
# Agent pushing a single movable object into wall and getting stopped.
# Agent pushing a single movable object into another movable object and pushing both of them, up, down, left, right.
# Agent pushing a single movable object into another movable object into a wall and getting stopped.


# Define test cases - comprehensive coverage of all scenarios
TEST_CASES = [
    # 1. Agent moving left, right, up, down normally with no wall
    TestCase(
        name="agent_move_left_basic",
        initial_puzzle_path="agent_move_basic.pwp",
        final_puzzle_path="agent_move_basic_left.pwp",
        action=0,  # LEFT
        description="Agent moves left with no obstacles",
    ),
    TestCase(
        name="agent_move_right_basic",
        initial_puzzle_path="agent_move_basic.pwp",
        final_puzzle_path="agent_move_basic_right.pwp",
        action=1,  # RIGHT
        description="Agent moves right with no obstacles",
    ),
    TestCase(
        name="agent_move_up_basic",
        initial_puzzle_path="agent_move_basic.pwp",
        final_puzzle_path="agent_move_basic_up.pwp",
        action=2,  # UP
        description="Agent moves up with no obstacles",
    ),
    TestCase(
        name="agent_move_down_basic",
        initial_puzzle_path="agent_move_basic.pwp",
        final_puzzle_path="agent_move_basic_down.pwp",
        action=3,  # DOWN
        description="Agent moves down with no obstacles",
    ),
    # 2. Agent moving into puzzle boundary when there's no wall
    TestCase(
        name="agent_boundary_blocked_left",
        initial_puzzle_path="boundary_test.pwp",
        final_puzzle_path="boundary_test.pwp",  # Same - no movement expected
        action=0,  # LEFT
        description="Agent blocked by boundary when moving left",
    ),
    # 3. Agent moving into wall and getting stopped
    TestCase(
        name="agent_blocked_by_wall_right",
        initial_puzzle_path="wall_blocked.pwp",
        final_puzzle_path="wall_blocked.pwp",  # Same - no movement expected
        action=1,  # RIGHT
        description="Agent blocked by wall, no movement",
    ),
    TestCase(
        name="agent_blocked_by_wall_left",
        initial_puzzle_path="wall_blocked.pwp",
        final_puzzle_path="wall_blocked.pwp",  # Same - no movement expected
        action=0,  # LEFT
        description="Agent blocked by wall, no movement",
    ),
    TestCase(
        name="agent_blocked_by_wall_up",
        initial_puzzle_path="wall_blocked.pwp",
        final_puzzle_path="wall_blocked.pwp",  # Same - no movement expected
        action=2,  # UP
        description="Agent blocked by wall, no movement",
    ),
    TestCase(
        name="agent_blocked_by_wall_down",
        initial_puzzle_path="wall_blocked.pwp",
        final_puzzle_path="wall_blocked.pwp",  # Same - no movement expected
        action=3,  # DOWN
        description="Agent blocked by wall, no movement",
    ),
    # 4. Agent pushing a single movable object up, down, left, right
    TestCase(
        name="push_single_object_right",
        initial_puzzle_path="push_single_object.pwp",
        final_puzzle_path="push_single_object_right.pwp",
        action=1,  # RIGHT
        description="Agent pushes single movable object right",
    ),
    TestCase(
        name="push_single_object_left",
        initial_puzzle_path="push_single_object.pwp",
        final_puzzle_path="push_single_object_left.pwp",
        action=0,  # LEFT
        description="Agent pushes single movable object left",
    ),
    TestCase(
        name="push_single_object_up",
        initial_puzzle_path="push_single_object.pwp",
        final_puzzle_path="push_single_object_up.pwp",
        action=2,  # UP
        description="Agent pushes single movable object up",
    ),
    TestCase(
        name="push_single_object_down",
        initial_puzzle_path="push_single_object.pwp",
        final_puzzle_path="push_single_object_down.pwp",
        action=3,  # DOWN
        description="Agent pushes single movable object down",
    ),
    # 5. Agent pushing a single movable object into wall and getting stopped
    TestCase(
        name="push_object_into_wall_blocked",
        initial_puzzle_path="push_object_blocked.pwp",
        final_puzzle_path="push_object_blocked.pwp",  # Same - no movement expected
        action=1,  # RIGHT
        description="Agent tries to push object into wall, gets stopped",
    ),
    # 6. Agent pushing a single movable object into another movable object and pushing both
    TestCase(
        name="push_two_objects_right",
        initial_puzzle_path="push_two_objects_right.pwp",
        final_puzzle_path="push_two_objects_right_right.pwp",
        action=1,  # RIGHT
        description="Agent pushes one object into another, moving both right",
    ),
    TestCase(
        name="push_two_objects_left",
        initial_puzzle_path="push_two_objects_left.pwp",
        final_puzzle_path="push_two_objects_left_left.pwp",  # Should move left freely
        action=0,  # LEFT
        description="Agent pushes one object into another, moving both left",
    ),
    TestCase(
        name="push_two_objects_up",
        initial_puzzle_path="push_two_objects_up.pwp",
        final_puzzle_path="push_two_objects_up_up.pwp",
        action=2,  # UP
        description="Agent pushes one object into another, moving both up",
    ),
    TestCase(
        name="push_two_objects_down",
        initial_puzzle_path="push_two_objects_down.pwp",
        final_puzzle_path="push_two_objects_down_down.pwp",
        action=3,  # DOWN
        description="Agent pushes one object into another, moving both down",
    ),
    # 7. Agent pushing movable object into another movable object into a wall (blocked)
    TestCase(
        name="push_two_objects_into_wall_blocked",
        initial_puzzle_path="push_two_objects_blocked.pwp",
        final_puzzle_path="push_two_objects_blocked.pwp",  # Same - no movement expected
        action=1,  # RIGHT
        description="Agent tries to push two objects into wall, gets stopped",
    ),
    # Now we test a variety of complex puzzles
    # Complex puzzle 1
    TestCase(
        name="complex_puzzle_1_left",
        initial_puzzle_path="complex_puzzle_1.pwp",
        final_puzzle_path="complex_puzzle_1_left.pwp",
        action=0,  # LEFT
        description="Agent pushes complex puzzle 1 left",
    ),
    TestCase(
        name="complex_puzzle_1_down",
        initial_puzzle_path="complex_puzzle_1.pwp",
        final_puzzle_path="complex_puzzle_1_down.pwp",
        action=3,  # DOWN
        description="Agent pushes complex puzzle 1 down",
    ),
    # Complex puzzle 2
    TestCase(
        name="complex_puzzle_2_left",
        initial_puzzle_path="complex_puzzle_2.pwp",
        final_puzzle_path="complex_puzzle_2_left.pwp",
        action=0,  # LEFT
        description="Agent pushes complex puzzle 2 left",
    ),
]


@pytest.fixture(params=TEST_CASES, ids=lambda tc: tc.name)
def test_case(request):
    """Parametrized fixture that loads each test case."""
    test_case_def = request.param
    test_dir = Path(__file__).parent / "test_puzzles"

    initial_path = test_dir / test_case_def.initial_puzzle_path
    final_path = test_dir / test_case_def.final_puzzle_path

    # Skip if files don't exist
    if not initial_path.exists():
        pytest.skip(f"Initial puzzle file {initial_path} not found")
    if not final_path.exists():
        pytest.skip(f"Final puzzle file {final_path} not found")

    # Load initial state
    initial_puzzle, initial_state, observation = parse_puzzle_file(str(initial_path))

    # Load expected final state
    expected_puzzle, _, _ = parse_puzzle_file(str(final_path))

    return {
        "name": test_case_def.name,
        "description": test_case_def.description,
        "initial_puzzle": initial_puzzle,
        "initial_state": initial_state,
        "observation": observation,
        "expected_puzzle": expected_puzzle,
        "action": test_case_def.action,
    }


@pytest.fixture
def simple_test_case():
    """Simple fixture for manual test case specification."""

    def _create_test_case(initial_file: str, final_file: str, action: int):
        test_dir = Path(__file__).parent / "test_puzzles"

        initial_path = test_dir / initial_file
        final_path = test_dir / final_file

        if not initial_path.exists() or not final_path.exists():
            pytest.skip(f"Puzzle files not found: {initial_file} or {final_file}")

        initial_puzzle, initial_state, observation = parse_puzzle_file(str(initial_path))
        expected_puzzle, _, _ = parse_puzzle_file(str(final_path))

        return initial_puzzle, initial_state, observation, expected_puzzle, action

    return _create_test_case


class TestActionCorrectness:
    """Test actions using initial/final puzzle file pairs."""

    def test_action_produces_expected_result(self, test_case):
        """Generic test that verifies action produces expected final state."""
        print(f"\n🔄 Testing: {test_case['name']}")
        print(f"📝 Description: {test_case['description']}")
        print(f"🎮 Action: {test_case['action']}")

        # Get test data
        initial_state = test_case["initial_state"]
        observation = test_case["observation"]
        expected_puzzle = test_case["expected_puzzle"]
        action = test_case["action"]

        # Apply action
        take_action_all_jit = jax.jit(take_action_all)
        result_state = take_action_all_jit(observation, initial_state, action)

        # Compare final state with expected
        assert compare_states(result_state, expected_puzzle, test_case["name"]), (
            f"Final state doesn't match expected state for test: {test_case['name']}"
        )

    @pytest.mark.skip(reason="Manual test case for special scenarios")
    def test_manual_case_agent_left(self, simple_test_case):
        """Example of manual test case specification."""
        initial_puzzle, initial_state, observation, expected_puzzle, action = simple_test_case(
            "test_puzzle.pwp",
            "test_puzzle_move_left_expected.pwp",
            0,  # LEFT
        )

        # Apply action
        result_state = take_action_all(observation, initial_state, action)

        # Verify result
        assert compare_states(result_state, expected_puzzle, "manual_agent_left")


class TestHelperFunctions:
    """Test helper functions for debugging."""

    def test_state_comparison_function(self):
        """Test the state comparison utility."""
        test_dir = Path(__file__).parent / "test_puzzles"
        puzzle_path = test_dir / "test_puzzle.pwp"

        if not puzzle_path.exists():
            pytest.skip("test_puzzle.pwp not found")

        puzzle1, state1, _ = parse_puzzle_file(str(puzzle_path))
        puzzle2, state2, _ = parse_puzzle_file(str(puzzle_path))

        # Should match (same file)
        assert compare_states(state1, puzzle2, "identical_states")

    def test_puzzle_loading(self):
        """Test that puzzle files can be loaded."""
        test_dir = Path(__file__).parent / "test_puzzles"
        puzzle_files = list(test_dir.glob("*.pwp"))

        assert len(puzzle_files) > 0, "No puzzle files found"

        for puzzle_file in puzzle_files:
            puzzle, state, observation = parse_puzzle_file(str(puzzle_file))
            assert observation.shape == (LEVEL0_ALL_SIZE, LEVEL0_ALL_SIZE, 8)
            print(f"✅ Loaded {puzzle_file.name}")


# Run tests with:
# pytest test_actions_all.py::TestActionCorrectness -v -s
# pytest test_actions_all.py::TestActionCorrectness::test_action_produces_expected_result -v -s
