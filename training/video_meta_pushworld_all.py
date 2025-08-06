import os

import imageio
import jax
import jax.numpy as jnp
from flax import struct

import xminigrid.envs.pushworld as pushworld
from xminigrid.envs.pushworld.benchmarks import load_all_benchmark
from xminigrid.envs.pushworld.constants import LEVEL0_ALL_SIZE, SUCCESS_REWARD
from xminigrid.envs.pushworld.environment import Environment, EnvParams
from xminigrid.envs.pushworld.envs.meta_task_all_pushworld import MetaTaskPushWorldEnvironmentAll
from xminigrid.envs.pushworld.nn import ActorCriticRNN
from xminigrid.envs.pushworld.types import PushWorldPuzzleAll
from xminigrid.envs.pushworld.wrappers import GoalObservationWrapper, GymAutoResetWrapper, Wrapper


# for tracking per-episode statistics during meta-RL evaluation
class VideoMetaRolloutStats(struct.PyTreeNode):
    episode_rewards: jax.Array  # Shape: [max_episodes] - reward for each episode
    episode_lengths: jax.Array  # Shape: [max_episodes] - length of each episode
    episode_solved: jax.Array  # Shape: [max_episodes] - whether each episode was solved
    total_reward: jax.Array  # Scalar - total reward across all episodes
    num_episodes_completed: jax.Array  # Scalar - number of episodes actually completed
    video_frames: (
        jax.Array
    )  # Number of frames should be num_consecutive_episodes * 100, since each episode is 100 steps.
    # Therefore shape should be [max_episodes * 100, H, W, 3]


COLOR_MAP = jnp.array(
    [
        [255, 255, 255],  # 0: White (background)
        [0, 255, 0],  # 1: Green (agent)
        [0, 0, 255],  # 2: Blue (movable)
        [255, 0, 0],  # 3: Red (movable_goal)
        [0, 0, 0],  # 4: Black (wall)
        [255, 127, 127],  # 5: Light Red (empty goal)
        [255, 0, 0],  # 6: Red (filled goal - same as movable_goal)
    ],
    dtype=jnp.uint8,
)


def obs_to_rgb(observation: jax.Array) -> jax.Array:
    """
    Renders a grid world observation into an RGB image using JAX operations.

    Args:
        observation: A jax.Array of shape (H, W, 8) representing the grid state.

    Returns:
        A jax.Array of shape (H*64, W*64, 3) representing the upscaled RGB image.
    """
    h, w, _ = observation.shape

    # Channel indices for clarity
    AGENT_CH = 0
    M1_CH, M2_CH, M3_CH, M4_CH = 1, 2, 3, 4
    G1_CH, G2_CH = 5, 6
    WALL_CH = 7

    # --- 1. Create a base canvas ---
    # Start with a white background.
    # The shape is (H, W), and each element is an index into the COLOR_MAP.
    canvas = jnp.zeros((h, w), dtype=jnp.int32)

    # --- 2. Layer the static objects ---
    # jnp.where(condition, value_if_true, value_if_false)
    # Layer walls
    wall_mask = observation[:, :, WALL_CH] > 0
    canvas = jnp.where(wall_mask, 4, canvas)  # 4 is the index for WALL color

    # Layer goals (as empty goals)
    g1_mask = observation[:, :, G1_CH] > 0
    g2_mask = observation[:, :, G2_CH] > 0
    canvas = jnp.where(g1_mask, 5, canvas)  # 5 is the index for GOAL_EMPTY color
    canvas = jnp.where(g2_mask, 5, canvas)

    # --- 3. Determine movable colors based on goal existence ---
    # This avoids Python-level if/else statements.
    g1_exists = jnp.any(g1_mask)
    g2_exists = jnp.any(g2_mask)

    # Color for m1 is always 'movable_goal' as g1 always exists.
    m1_color_idx = 3  # movable_goal (red)

    # Color for m2 depends on whether g2 exists.
    # jnp.where works on scalars too, making it great for conditional logic.
    m2_color_idx = jnp.where(
        g2_exists,
        3,  # movable_goal (red) if g2 exists
        2,
    )  # movable (blue) if g2 does not exist

    # m3 and m4 are always regular 'movable'.
    m3_color_idx = 2  # movable (blue)
    m4_color_idx = 2  # movable (blue)

    # --- 4. Layer the movables ---
    m1_mask = observation[:, :, M1_CH] > 0
    m2_mask = observation[:, :, M2_CH] > 0
    m3_mask = observation[:, :, M3_CH] > 0
    m4_mask = observation[:, :, M4_CH] > 0

    canvas = jnp.where(m1_mask, m1_color_idx, canvas)
    canvas = jnp.where(m2_mask, m2_color_idx, canvas)
    canvas = jnp.where(m3_mask, m3_color_idx, canvas)
    canvas = jnp.where(m4_mask, m4_color_idx, canvas)

    # --- 5. Handle filled goals ---
    # If a movable is on a goal, the color should be the 'filled goal' color.
    # This is an overlay operation.
    m1_on_g1 = m1_mask & g1_mask
    m2_on_g2 = m2_mask & g2_mask
    canvas = jnp.where(m1_on_g1, 6, canvas)  # 6 is the index for GOAL_FILLED
    canvas = jnp.where(g2_exists, jnp.where(m2_on_g2, 6, canvas), canvas)

    # --- 6. Layer the agent on top ---
    agent_mask = observation[:, :, AGENT_CH] > 0
    canvas = jnp.where(agent_mask, 1, canvas)  # 1 is the index for AGENT color

    # --- 7. Convert the canvas of indices to an RGB image ---
    # This is a powerful indexing operation in JAX.
    rgb_img = COLOR_MAP[canvas]

    # --- 8. Upscale for better visibility ---
    # upscaled_img = jnp.kron(rgb_img, jnp.ones((64, 64, 1), dtype=jnp.uint8))

    return rgb_img


def meta_rollout_video(
    rng: jax.Array,
    env: Environment,
    env_params: EnvParams,
    eval_puzzle: PushWorldPuzzleAll,
    params,
    network: ActorCriticRNN,
    init_hstate: jax.Array,
    num_consecutive_episodes: int = 1,
) -> VideoMetaRolloutStats:
    """Rollout that tracks statistics for each individual episode."""

    def _reset_env_fn(carry_components):
        """Function to execute when the episode has ended."""
        timestep, env_params, rng = carry_components
        key, _ = jax.random.split(rng)
        reset_timestep = env.reset(env_params, key)

        # Return a new timestep with the reset state and observation
        return timestep.replace(
            state=reset_timestep.state,
            observation=reset_timestep.observation,
        )

    def _identity_fn(carry_components):
        """Function to execute when the episode has not ended."""
        timestep, _, _ = carry_components
        # Return the timestep unchanged
        return timestep

    def _cond_fn(carry):
        (
            rng,
            stats,
            timestep,
            prev_action,
            prev_reward,
            hstate,
            current_episode_reward,
            current_episode_length,
            step_num,
        ) = carry
        return jnp.less(stats.num_episodes_completed, num_consecutive_episodes)

    def _body_fn(carry):
        (
            rng,
            stats,
            timestep,
            prev_action,
            prev_reward,
            hstate,
            current_episode_reward,
            current_episode_length,
            step_num,
        ) = carry

        rng, _rng = jax.random.split(rng)
        dist, _, hstate = network.apply(
            params,
            {
                # We add single channel dimension to end of obs_img
                "obs": timestep.observation[None, None, ...],
                "prev_action": prev_action[None, None, ...],
                "prev_reward": prev_reward[None, None, ...],
            },
            hstate,
        )
        action = dist.sample(seed=_rng).squeeze()
        timestep = env.step(env_params, timestep, action)

        # Update current episode accumulators
        current_episode_reward = current_episode_reward + timestep.reward
        current_episode_length = current_episode_length + 1

        # Check if episode ended
        episode_ended = timestep.last()
        solved_flag = ((timestep.reward == SUCCESS_REWARD) & (episode_ended == 1)).astype(jnp.int32)

        # When episode ends, store the episode stats
        episode_idx = stats.num_episodes_completed
        new_episode_rewards = stats.episode_rewards.at[episode_idx].set(
            jnp.where(episode_ended, current_episode_reward, stats.episode_rewards[episode_idx])
        )
        new_episode_lengths = stats.episode_lengths.at[episode_idx].set(
            jnp.where(episode_ended, current_episode_length, stats.episode_lengths[episode_idx])
        )
        new_episode_solved = stats.episode_solved.at[episode_idx].set(
            jnp.where(episode_ended, solved_flag, stats.episode_solved[episode_idx])
        )

        # Update stats
        stats = stats.replace(
            episode_rewards=new_episode_rewards,
            episode_lengths=new_episode_lengths,
            episode_solved=new_episode_solved,
            total_reward=stats.total_reward + timestep.reward,
            num_episodes_completed=stats.num_episodes_completed + episode_ended,
            video_frames=stats.video_frames.at[step_num].set(obs_to_rgb(timestep.observation)),
        )

        # Reset episode accumulators when episode ends
        current_episode_reward = jnp.where(episode_ended, 0.0, current_episode_reward)
        current_episode_length = jnp.where(episode_ended, 0, current_episode_length)

        operands = (timestep, env_params, rng)
        timestep = jax.lax.cond(
            episode_ended,
            _reset_env_fn,  # The function to run if True
            _identity_fn,  # The function to run if False
            operands,  # The arguments passed to either function
        )

        carry = (
            rng,
            stats,
            timestep,
            action,
            timestep.reward,
            hstate,
            current_episode_reward,
            current_episode_length,
            step_num + 1,
        )
        return carry

    # Initialize episode tracking arrays
    episode_rewards = jnp.zeros(num_consecutive_episodes)
    episode_lengths = jnp.zeros(num_consecutive_episodes, dtype=jnp.int32)
    episode_solved = jnp.zeros(num_consecutive_episodes, dtype=jnp.int32)

    initial_stats = VideoMetaRolloutStats(
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        episode_solved=episode_solved,
        total_reward=jnp.asarray(0.0),
        num_episodes_completed=jnp.asarray(0),
        video_frames=jnp.zeros((num_consecutive_episodes * 100, LEVEL0_ALL_SIZE, LEVEL0_ALL_SIZE, 3), dtype=jnp.uint8),
    )

    env_params = env_params.replace(puzzle=eval_puzzle)
    timestep = env.eval_reset(env_params, rng)
    prev_action = jnp.asarray(0)
    prev_reward = jnp.asarray(0)
    current_episode_reward = jnp.asarray(0.0)
    current_episode_length = jnp.asarray(0)

    init_carry = (
        rng,
        initial_stats,
        timestep,
        prev_action,
        prev_reward,
        init_hstate,
        current_episode_reward,
        current_episode_length,
        jnp.asarray(0),
    )

    final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    return final_carry[1]


# Set up environment and run rollouts to get videos
def get_videos_for_all(config, model_params):
    # Get model params from W&B
    env = MetaTaskPushWorldEnvironmentAll()
    env_params = env.default_params()
    # We want to include the final goal-reaching frame
    # env = GymAutoResetWrapper(env)

    benchmark = pushworld.load_all_benchmark(config.benchmark_id)

    puzzle_rng = jax.random.key(config.puzzle_seed)
    train_rng, test_rng = jax.random.split(puzzle_rng)

    if config.num_train is not None:
        assert config.num_train <= benchmark.num_train_puzzles(), (
            "num_train is larger than num train available in benchmark"
        )
        perm = jax.random.permutation(train_rng, benchmark.num_train_puzzles())
        idxs = perm[: config.num_train]
        benchmark = benchmark.replace(train_puzzles=benchmark.train_puzzles[idxs])
    else:
        config.num_train = benchmark.num_train_puzzles()

    if config.num_test is not None:
        assert config.num_test <= benchmark.num_test_puzzles(), (
            "num_test is larger than num test available in benchmark"
        )
        perm = jax.random.permutation(test_rng, benchmark.num_test_puzzles())
        idxs = perm[: config.num_test]
        benchmark = benchmark.replace(test_puzzles=benchmark.test_puzzles[idxs])
    else:
        config.num_test = benchmark.num_test_puzzles()

    if config.train_test_same:
        benchmark = benchmark.replace(test_puzzles=benchmark.train_puzzles)
        config.num_test = config.num_train

    network = ActorCriticRNN(
        num_actions=env.num_actions(env_params),
        obs_emb_dim=config.obs_emb_dim,
        action_emb_dim=config.action_emb_dim,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_num_layers=config.rnn_num_layers,
        head_hidden_dim=config.head_hidden_dim,
        img_obs=config.img_obs,
        dtype=jnp.bfloat16 if config.enable_bf16 else None,
    )

    init_hstate = network.initialize_carry(batch_size=config.num_envs_per_device)
    eval_hstate = init_hstate[0][None]

    eval_reset_rng = jax.random.key(config.eval_seed)
    eval_test_rng, eval_train_rng = jax.random.split(eval_reset_rng)

    eval_test_reset_rng = jax.random.split(eval_test_rng, num=config.num_test)
    eval_test_puzzles = benchmark.get_test_puzzles()
    eval_test_stats = jax.vmap(meta_rollout_video, in_axes=(0, None, None, 0, None, None, None, None))(
        eval_test_reset_rng,
        env,
        env_params,
        eval_test_puzzles,
        model_params,
        network,
        eval_hstate,
        config.eval_num_episodes,
    )
    # eval_test_stats = jax.lax.pmean(eval_test_stats, axis_name="devices")

    # Eval on train set
    eval_train_reset_rng = jax.random.split(eval_train_rng, num=config.num_train)
    eval_train_puzzles = benchmark.get_train_puzzles()
    eval_train_stats = jax.vmap(meta_rollout_video, in_axes=(0, None, None, 0, None, None, None, None))(
        eval_train_reset_rng,
        env,
        env_params,
        eval_train_puzzles,
        model_params,
        network,
        eval_hstate,
        config.eval_num_episodes,
    )
    # eval_train_stats = jax.lax.pmean(eval_train_stats, axis_name="devices")

    return eval_test_stats, eval_train_stats


def upscale_and_save(videos, ids, save_path, name, batch_size=10):
    """
    Upscales and saves a collection of videos to disk in batches.

    This function processes videos in small batches to avoid high memory usage
    when dealing with large datasets. Each frame is upscaled, and the resulting
    video is saved as an MP4.

    Args:
        videos (jax.Array): A JAX array of video frames.
        ids (list | jax.Array): A list or array of identifiers for each video, used for naming files.
        save_path (str): The directory where the output MP4 files will be saved.
        name (str): A prefix used for the output filenames (e.g., 'train_success').
        batch_size (int): The number of videos to process in each batch.
    """
    os.makedirs(save_path, exist_ok=True)

    # 1. Upscale
    def upscale_frame(frame):
        """Upscales a single image frame."""
        scale_factor = 16
        upscale_matrix = jnp.ones((scale_factor, scale_factor, 1), dtype=jnp.uint8)
        return jnp.kron(frame, upscale_matrix)

    vmapped_upscaler = jax.jit(jax.vmap(upscale_frame))

    # --- 2. The Batching Loop ---
    num_videos = len(ids)
    for i in range(0, num_videos, batch_size):
        # Determine the start and end index for the current batch
        start_idx = i
        end_idx = min(i + batch_size, num_videos)

        print(f"Processing batch {i // batch_size + 1}: Videos {start_idx} to {end_idx - 1}")

        # Get the current batch of videos and IDs
        video_batch = videos[start_idx:end_idx]
        id_batch = ids[start_idx:end_idx]

        # --- 3. Run JAX computation ONLY on the small batch ---

        # This now only allocates memory for one upscaled batch, not the whole dataset
        upscaled_batch = vmapped_upscaler(video_batch)

        # Block until computation is finished before saving
        upscaled_batch.block_until_ready()

        # --- 4. Save the processed batch to disk ---
        for j, video_id in enumerate(id_batch):
            # Get the single video from the processed batch
            video_data = upscaled_batch[j]
            file_path = os.path.join(save_path, f"{name}_{video_id}.mp4")
            imageio.mimsave(file_path, video_data, fps=16, format="mp4")

    print("All batches processed and saved successfully.")


def upscale_and_save_videos(
    train_stats,
    test_stats,
    video_save_root,
    max_train_success=100,
    max_train_fail=100,
    max_test_success=None,
    max_test_fail=100,
):
    """
    Sorts videos from training and testing sets by outcome and saves them.

    This function filters videos into 'succeeded' and 'failed' categories based on
    the final solved status in the provided statistics. It then calls a utility
    to process and save them into separate subdirectories (e.g., 'train/succeeded',
    'test/failed').

    Args:
        train_stats: An object containing `video_frames` and `episode_solved`
                     data for the training set.
        test_stats: An object containing `video_frames` and `episode_solved`
                    data for the testing set.
        video_save_root (str): The root directory where the categorized video
                         folders will be created.
    """
    train_succeed_idx = jnp.where(train_stats.episode_solved[:, -1] == 1)[0]
    train_fail_idx = jnp.where(train_stats.episode_solved[:, -1] == 0)[0]
    test_succeed_idx = jnp.where(test_stats.episode_solved[:, -1] == 1)[0]
    test_fail_idx = jnp.where(test_stats.episode_solved[:, -1] == 0)[0]

    upscale_and_save(
        train_stats.video_frames[train_succeed_idx][:max_train_success],
        train_succeed_idx[:max_train_success],
        os.path.join(video_save_root, "train", "succeeded"),
        "train_success",
    )

    upscale_and_save(
        train_stats.video_frames[train_fail_idx][:max_train_fail],
        train_fail_idx[:max_train_fail],
        os.path.join(video_save_root, "train", "failed"),
        "train_fail",
    )

    upscale_and_save(
        test_stats.video_frames[test_succeed_idx][:max_test_success],
        test_succeed_idx[:max_test_success],
        os.path.join(video_save_root, "test", "succeeded"),
        "test_success",
    )

    upscale_and_save(
        test_stats.video_frames[test_fail_idx][:max_test_fail],
        test_fail_idx[:max_test_fail],
        os.path.join(video_save_root, "test", "failed"),
        "test_fail",
    )
