import jax
import jax.lax as lax
import jax.numpy as jnp

from .constants import LEVEL0_ALL_SIZE
from .types import GridState, IntOrArray, StateAll

DISPLACEMENTS = jnp.array(
    [
        (-1, 0),  # LEFT
        (1, 0),  # RIGHT
        (0, -1),  # UP
        (0, 1),  # DOWN
    ]
)

ID_TO_CHANNEL = {
    "a": 0,
    "m1": 1,
    "m2": 2,
    "m3": 3,
    "m4": 4,
    "g1": 5,
    "g2": 6,
    "w": 7,
}


def take_action_all(observation: jax.Array, state: StateAll, action: IntOrArray) -> StateAll:
    # observation: one-hot encoding of all the different objects in the grid. Stack of 8 channels.

    displacement = DISPLACEMENTS[action]
    return move_jax_masked(observation, state, displacement)


def masked_displacement(
    coords: jnp.ndarray,  # (max_pix, 2)
    displacement: jnp.ndarray,  # (2,)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    For each row in coords:
      valid_mask[i] = True  if coords[i] != (-1,-1)
      disp_all[i]   = coords[i] + displacement
      disp[i]       = disp_all[i]     if valid_mask[i]
                     = coords[i]       otherwise
    """
    valid = (coords[:, 0] >= 0) & (coords[:, 1] >= 0)  # (max_pix,)
    disp_all = coords + displacement  # (max_pix,2)
    disp = jnp.where(valid[:, None], disp_all, coords)  # (max_pix,2)
    return disp, valid


def move_jax_masked(observation, state: StateAll, displacement):
    # stack coords for agent + 4 movable objects
    coords = jnp.stack([state.a, state.m1, state.m2, state.m3, state.m4], axis=0)  # (N=5, max_pix, 2)
    N = coords.shape[0]

    walls = observation[..., ID_TO_CHANNEL["w"]]

    # frontier mask: start with only the agent (idx 0) in the wavefront
    frontier = jnp.array([True] + [False] * (N - 1))
    # pushed mask: mark agent as already "pushed" so we don't revisit it
    pushed = jnp.zeros((N,), dtype=bool).at[0].set(True)
    # broken flag: set to True if any frontier step hits wall/OOB
    broken = False

    def cond_fn(carry):
        frontier, pushed, broken = carry
        # keep going while there is at least one index in the frontier
        return frontier.any()

    # compute one-step displacements + valid-pixel mask for every object
    all_disp, all_valid = jax.vmap(masked_displacement, in_axes=(0, None))(coords, displacement)
    # all_disp:  (N, max_pix, 2)
    # all_valid: (N, max_pix)

    def body_fn(carry):
        frontier, pushed, broken = carry

        # --- detect any wall or OOB hits for objects in the frontier ---
        ys = all_disp[..., 1]
        xs = all_disp[..., 0]
        raw_vals = walls[ys, xs]  # (N, max_pix)
        wall_vals = jnp.where(all_valid, raw_vals, 0)
        hit_wall = jnp.any(wall_vals == 1, axis=1)  # (N,)

        inb = (xs >= 0) & (xs < walls.shape[1]) & (ys >= 0) & (ys < walls.shape[0])
        valid_inb = jnp.where(all_valid, inb, True)
        oob = ~jnp.all(valid_inb, axis=1)  # (N,)

        blocked = hit_wall | oob  # (N,)
        # if any frontier object is blocked, kill the wavefront entirely
        blocked_any = jnp.any(blocked & frontier)

        # record that we "broke" if the agent (frontier[0]) hit something
        broken = broken | blocked_any

        # --- compute collision graph: which objects touch which after move ---
        # collision-graph: for each i,j, do any pixels match?
        # — broadcast all_disp vs coords to compare ALL pixel combinations —
        eq = all_disp[:, :, None, None, :] == coords[None, None, :, :, :]
        # eq: (N, max_pix, N, max_pix, 2)
        # eq[i, p_i, j, p_j, coord]: Does moving object i's pixel p_i match stationary object j's pixel p_j?

        pixel_eq = jnp.all(eq, axis=-1)  # (N, max_pix, N, max_pix) - both x and y match

        # Mask out invalid pixels: only consider collisions between valid pixels
        # all_valid: (N, max_pix) - which pixels are valid for each object
        valid_i = all_valid[:, :, None, None]  # (N, max_pix, 1, 1) - valid pixels for moving objects
        valid_j = all_valid[None, None, :, :]  # (1, 1, N, max_pix) - valid pixels for stationary objects

        # Only count collision if BOTH pixels are valid, this is to make sure we don't count
        # collisions between sentinel pixels ((-1, -1)).
        valid_collision = pixel_eq & valid_i & valid_j  # (N, max_pix, N, max_pix)

        # Check if ANY pixel combination results in collision
        coll_mat = jnp.any(valid_collision, axis=(1, 3))  # (N, N)

        # any neighbor collisions coming out of the current frontier
        frontier_mat = frontier[:, None]  # (N, 1)
        neighbors = jnp.any(coll_mat * frontier_mat, axis=0)  # (N,)

        # new wavefront: hit neighbors we haven't yet pushed
        new_frontier = neighbors & (~pushed)
        pushed = pushed | new_frontier

        # if we didn't block this iteration, frontier moves on
        frontier = jnp.where(blocked_any, jnp.zeros_like(frontier), new_frontier)

        return frontier, pushed, broken

    # run the masked-wavefront until no new hits
    final_frontier, final_pushed, final_broken = lax.while_loop(cond_fn, body_fn, (frontier, pushed, broken))

    # For debugging:
    # carry = (frontier, pushed, broken)
    # while cond_fn(carry):
    #     carry = body_fn(carry)
    # final_frontier, final_pushed, final_broken = carry

    # agent moved OK only if it was ever "pushed" AND we never "broke"
    moved_ok = final_pushed[0] & (~final_broken)

    # apply the push to all objects in one shot
    def do_push(st):
        # Only move objects that are part of the push chain
        # (masked_displacement already handles invalid pixels correctly)
        should_move = final_pushed[:, None]  # (N, max_pix)
        moved = jnp.where(should_move[:, :, None], all_disp, coords)

        return st.replace(
            a=moved[0],
            m1=moved[1],
            m2=moved[2],
            m3=moved[3],
            m4=moved[4],
        )

    # if the agent (idx 0) made it through, perform the move; otherwise leave state unchanged
    return lax.cond(moved_ok, do_push, lambda st: st, state)

    # For debugging:
    # if moved_ok:
    #     return do_push(state)
    # else:
    #     return state


def num_goals_reached(observation: jax.Array) -> jax.Array:
    """Count how many goals are reached (objects on their respective goals)."""
    # Define object-goal pairs to check
    obj_goal_pairs = [("m1", "g1"), ("m2", "g2")]

    goals_count = jnp.array(0)  # Start with JAX array instead of Python int
    for obj, goal in obj_goal_pairs:
        obj_channel = observation[..., ID_TO_CHANNEL[obj]]
        goal_channel = observation[..., ID_TO_CHANNEL[goal]]

        # Check if goal exists (not all zeros) and object matches goal exactly
        has_goal = jnp.any(goal_channel)
        on_goal = jnp.array_equal(obj_channel, goal_channel)
        goals_count += jnp.where(has_goal, on_goal, False)

    return goals_count


def count_total_goals(observation: jax.Array) -> jax.Array:
    """Count how many goals exist in the level."""
    goal_channels = ["g1", "g2"]
    total_goals = jnp.array(0)
    for goal in goal_channels:
        goal_channel = observation[..., ID_TO_CHANNEL[goal]]
        has_goal = jnp.any(goal_channel)
        total_goals += jnp.where(has_goal, 1, 0)
    return total_goals


def check_goal(observation: jax.Array, state: StateAll) -> jax.Array:
    """Check if all existing goals have been reached."""
    goals_reached = num_goals_reached(observation)
    total_goals = count_total_goals(observation)
    # All goals reached when the number of goals reached equals the total number of existing goals
    return goals_reached == total_goals
