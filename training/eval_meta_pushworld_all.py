import imageio
import jax
import jax.numpy as jnp
import numpy as np
from eval_utils import text_to_rgb_all
from nn_pushworld import ActorCriticRNN

from xminigrid.envs.pushworld.benchmarks import BenchmarkAll
from xminigrid.envs.pushworld.constants import Tiles
from xminigrid.envs.pushworld.envs.meta_task_pushworld_all import MetaTaskPushWorldEnvironmentAll
from xminigrid.envs.pushworld.wrappers import GymAutoResetWrapper


def evaluate_meta(train_info, config, puzzles, video_name, eval_seed):
    META_EPISODES = 10
    # We're only going to sample from test anyway
    benchmark = BenchmarkAll(
        train_puzzles=puzzles,
        test_puzzles=puzzles,
    )

    # setup environment
    env = MetaTaskPushWorldEnvironmentAll()
    env = GymAutoResetWrapper(env)
    env_params = env.default_params()

    rng = jax.random.key(eval_seed)
    rng, _rng = jax.random.split(rng)

    puzzle = benchmark.sample_puzzle(_rng, "test")
    env_params = env_params.replace(puzzle=puzzle)

    # you can use train_state from the final state also
    # we just demo here how to do it if you loaded params from the checkpoint
    params = train_info["state"].params
    model = ActorCriticRNN(
        num_actions=env.num_actions(env_params),
        action_emb_dim=config.action_emb_dim,
        rnn_hidden_dim=config.rnn_hidden_dim,
        rnn_num_layers=config.rnn_num_layers,
        head_hidden_dim=config.head_hidden_dim,
        img_obs=config.img_obs,
    )

    # jitting all functions
    apply_fn, reset_fn, step_fn = jax.jit(model.apply), jax.jit(env.reset), jax.jit(env.step)

    # for logging
    total_reward, num_episodes = 0, 0
    rendered_imgs = []

    rng, _rng = jax.random.split(rng)

    # initial inputs
    hidden = model.initialize_carry(1)
    prev_reward = jnp.asarray(0)
    prev_action = jnp.asarray(0)

    timestep = reset_fn(env_params, _rng)
    rendered_imgs.append(text_to_rgb_all(timestep.observation))

    while num_episodes < META_EPISODES:
        rng, _rng = jax.random.split(rng)
        dist, _, hidden = apply_fn(
            params,
            {
                "obs": timestep.observation[None, None, ...],
                "prev_action": prev_action[None, None, ...],
                "prev_reward": prev_reward[None, None, ...],
            },
            hidden,
        )
        action = dist.sample(seed=_rng).squeeze()

        timestep = step_fn(env_params, timestep, action)
        prev_action = action
        prev_reward = timestep.reward

        total_reward += timestep.reward.item()
        num_episodes += int(timestep.last().item())
        rendered_imgs.append(text_to_rgb_all(timestep.observation))

    imageio.mimsave(f"{video_name}.mp4", rendered_imgs, fps=16, format="mp4")
    return total_reward
