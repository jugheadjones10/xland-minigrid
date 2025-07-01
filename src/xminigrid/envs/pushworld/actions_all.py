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

    # frontier = [AGENT]

    displacement = DISPLACEMENTS[action]
    # return move(observation, state, displacement)
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


def move_jax(observation, state: StateAll, displacement):
    # stack coords for a, m1–m4
    coords = jnp.stack([state.a, state.m1, state.m2, state.m3, state.m4], axis=0)  # (N, max_pix, 2)
    N = coords.shape[0]  # 5 objects

    # frontier: indices 0…4; head/tail pointers
    frontier = jnp.arange(N)  # we'll only enqueue each idx once
    pushed = jnp.zeros((N,), bool).at[0].set(True)
    head, tail = 0, 1

    walls = observation[..., ID_TO_CHANNEL["w"]]

    def cond_fn(carry):
        head, tail, pushed = carry
        return head < tail

    def body_fn(carry):
        head, tail, pushed = carry
        obj_idx = frontier[head]
        this_coords = coords[obj_idx]  # (max_pix,2)

        # —— use masked_displacement here ——
        disp_coords, valid_mask = masked_displacement(this_coords, displacement)
        #
        # safe‐index into walls:
        ys = disp_coords[:, 1]
        xs = disp_coords[:, 0]
        safe_ys = jnp.where(valid_mask, ys, 0)
        safe_xs = jnp.where(valid_mask, xs, 0)
        raw_vals = walls[safe_ys, safe_xs]  # no OOB now
        wall_vals = jnp.where(valid_mask, raw_vals, 0)  # ignore sentinels
        hit_wall = jnp.any(wall_vals == 1)

        # bounds check (only on valid slots) - check both >= 0 and < shape
        inb = (safe_xs >= 0) & (safe_xs < walls.shape[1]) & (safe_ys >= 0) & (safe_ys < walls.shape[0])
        safe = jnp.where(valid_mask, inb, True)
        oob = ~jnp.all(safe)

        blocked = hit_wall | oob
        # if any blockage, short‐circuit by setting head=tail so loop ends
        tail = jnp.where(blocked, head, tail)

        # otherwise, find collisions with other objects
        # compare disp_coords (max_pix,2) vs coords (which becomes (N, max_pix, 2) after transposing)
        eq = disp_coords[:, None, :] == coords.transpose(1, 0, 2)  # (max_pix, N, 2)
        coll = jnp.any(eq.all(-1), axis=0)  # (N,)
        new_hits = coll & (~pushed)

        pushed = pushed | new_hits
        tail = tail + new_hits.sum()
        head += 1

        return (head, tail, pushed)

    # head, tail, pushed = lax.while_loop(cond_fn, body_fn, (head, tail, pushed))

    carry = (head, tail, pushed)
    while cond_fn(carry):
        carry = body_fn(carry)
    head, tail, pushed = carry

    # if blocked, return original; else apply the push
    def do_push(st):
        # apply_masked_displacement to each object in one shot
        all_disp, all_valid = jax.vmap(masked_displacement, in_axes=(0, None))(coords, displacement)
        # all_disp: (N, max_pix,2), all_valid: (N, max_pix)
        moved = jnp.where(all_valid[:, :, None], all_disp, coords)
        return st.replace(
            a=moved[0],
            m1=moved[1],
            m2=moved[2],
            m3=moved[3],
            m4=moved[4],
        )

    # we treated blockage by killing the loop early; just check pushed[0] to know if agent moved
    moved_ok = pushed[0]
    return lax.cond(moved_ok, do_push, lambda st: st, state)


def move_jax_masked(observation, state: StateAll, displacement):
    # stack coords for agent + 4 movable objects
    coords = jnp.stack([state.a, state.m1, state.m2, state.m3, state.m4], axis=0)  # (N=5, max_pix, 2)
    N = coords.shape[0]

    walls = observation[..., ID_TO_CHANNEL["w"]]

    # frontier mask: start with only the agent (idx 0) in the wavefront
    frontier = jnp.array([True] + [False] * (N - 1))
    # pushed mask: mark agent as already “pushed” so we don’t revisit it
    pushed = jnp.zeros((N,), dtype=bool).at[0].set(True)
    # broken flag: set to True if any frontier step hits wall/OOB
    broken = False

    def cond_fn(carry):
        frontier, pushed, broken = carry
        # keep going while there is at least one index in the frontier
        return frontier.any()

    # compute one‐step displacements + valid‐pixel mask for every object
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

        # record that we “broke” if the agent (frontier[0]) hit something
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

        # if we didn’t block this iteration, frontier moves on
        frontier = jnp.where(blocked_any, jnp.zeros_like(frontier), new_frontier)

        return frontier, pushed, broken

    # run the masked‐wavefront until no new hits
    final_frontier, final_pushed, final_broken = lax.while_loop(cond_fn, body_fn, (frontier, pushed, broken))

    # For debugging:
    # carry = (frontier, pushed, broken)
    # while cond_fn(carry):
    #     carry = body_fn(carry)
    # final_frontier, final_pushed, final_broken = carry

    # agent moved OK only if it was ever “pushed” AND we never “broke”
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
