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


def move(observation: jax.Array, state: StateAll, displacement: jax.Array) -> StateAll:
    frontier = ["a"]
    pushed = set()

    while len(frontier) > 0:
        movable_idx = frontier.pop()
        movable_coords = getattr(state, movable_idx)

        displaced_coords = []
        for coord in movable_coords:
            if jnp.all(coord != -1):
                new_coord = coord + displacement
                displaced_coords.append(new_coord)

        # Check if wall
        walls_channel = observation[..., ID_TO_CHANNEL["w"]]

        for coord in displaced_coords:
            if walls_channel[coord[1], coord[0]] == 1:
                # Transitive stopping; nothing can move
                return state

        # Check if any displaced coords are out of bounds
        for coord in displaced_coords:
            is_in_bounds = jnp.logical_and(
                # jnp.logical_and(coord[0] >= 0, coord[0] < LEVEL0_ALL_SIZE),
                # jnp.logical_and(coord[1] >= 0, coord[1] < LEVEL0_ALL_SIZE),
                jnp.logical_and(coord[1] >= 0, coord[1] < observation.shape[0]),
                jnp.logical_and(coord[0] >= 0, coord[0] < observation.shape[1]),
            )
            if not is_in_bounds:
                # Out of bounds; nothing can move
                return state

        # Check for other movables in the way
        for movable_idx in ["m1", "m2", "m3", "m4"]:
            if movable_idx in pushed:
                continue

            movable_channel = observation[..., ID_TO_CHANNEL[movable_idx]]

            # Check if any of the displaced coordinates collide with this movable
            if len(displaced_coords) > 0:
                # Convert list of coordinates to array for vectorized access
                coords_array = jnp.stack(displaced_coords)  # Shape: (n_coords, 2)
                x_coords = coords_array[:, 0]  # All x coordinates
                y_coords = coords_array[:, 1]  # All y coordinates

                # Check all positions at once using advanced indexing
                # Note: arrays are indexed as [y, x] since coords are [x, y] pairs
                values = movable_channel[y_coords, x_coords]
                if jnp.any(values == 1):
                    frontier.append(movable_idx)
                    pushed.add(movable_idx)

    # Apply displacements to all moved objects
    # If there was a wall or something in the way, we would have returned early.
    # The fact that we're here means that the agent can move.
    pushed.add("a")
    for moved in pushed:
        moved_coords = getattr(state, moved)
        for i, coord in enumerate(moved_coords):
            if jnp.all(coord != -1):
                new_coord = coord + displacement
                moved_coords = moved_coords.at[i].set(new_coord)
        setattr(state, moved, moved_coords)

    return state


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

    def cond_fn(carry):
        frontier, pushed = carry
        # keep going while there is at least one index in the frontier
        return frontier.any()

    def body_fn(carry):
        frontier, pushed = carry

        # compute one‐step displacements + valid‐pixel mask for every object
        all_disp, all_valid = jax.vmap(masked_displacement, in_axes=(0, None))(coords, displacement)
        # all_disp:  (N, max_pix, 2)
        # all_valid: (N, max_pix)

        # --- detect any wall or OOB hits for objects in the frontier ---
        ys = all_disp[..., 1]
        xs = all_disp[..., 0]
        safe_ys = jnp.where(all_valid, ys, 0)
        safe_xs = jnp.where(all_valid, xs, 0)
        raw_vals = walls[safe_ys, safe_xs]  # (N, max_pix)
        wall_vals = jnp.where(all_valid, raw_vals, 0)
        hit_wall = jnp.any(wall_vals == 1, axis=1)  # (N,)

        inb = (safe_xs >= 0) & (safe_xs < walls.shape[1]) & (safe_ys >= 0) & (safe_ys < walls.shape[0])
        oob = ~jnp.all(jnp.where(all_valid, inb, True), axis=1)  # (N,)

        blocked = hit_wall | oob  # (N,)
        # if any frontier object is blocked, kill the wavefront entirely
        blocked_any = jnp.any(blocked & frontier)
        frontier = jnp.where(blocked_any, jnp.zeros_like(frontier), frontier)

        # --- compute collision graph: which objects touch which after move ---
        # eq: (N, max_pix, 1, 2) == (1,   N,     max_pix, 2)
        eq = all_disp[:, :, None, :] == coords[None, :, :, :]
        # pixel_match: (N, max_pix, N), then reduce over pixels
        coll_mat = jnp.any(jnp.all(eq, axis=-1), axis=1)  # (N, N)

        # any neighbor collisions coming out of the current frontier
        neighbors = jnp.any(coll_mat * frontier[:, None], axis=0)  # (N,)

        # new wavefront: hit neighbors we haven't yet pushed
        new_frontier = neighbors & (~pushed)
        pushed = pushed | new_frontier

        # if blocked_any is True, we've already zeroed out frontier above
        frontier = new_frontier & (~blocked_any)

        return frontier, pushed

    # run the masked‐wavefront until no new hits
    final_frontier, final_pushed = lax.while_loop(cond_fn, body_fn, (frontier, pushed))
    moved_ok = final_pushed[0]

    # apply the push to all objects in one shot
    def do_push(st):
        all_disp, all_valid = jax.vmap(masked_displacement, in_axes=(0, None))(coords, displacement)
        moved = jnp.where(all_valid[:, :, None], all_disp, coords)
        return st.replace(
            a=moved[0],
            m1=moved[1],
            m2=moved[2],
            m3=moved[3],
            m4=moved[4],
        )

    # if the agent (idx 0) made it through, perform the move; otherwise leave state unchanged
    return lax.cond(moved_ok, do_push, lambda st: st, state)
