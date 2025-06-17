import imageio
import jax
import jax.numpy as jnp
import numpy as np
from eval_utils import text_to_rgb
from nn_pushworld import ActorCriticRNN

from xminigrid.envs.pushworld.benchmarks import Benchmark
from xminigrid.envs.pushworld.constants import Tiles
from xminigrid.envs.pushworld.envs.single_task_pushworld import SingleTaskPushWorldEnvironment
from xminigrid.envs.pushworld.wrappers import GoalObservationWrapper


def evaluate_single(train_info, config, puzzles, video_name, eval_seed):
    # We're only going to sample from test anyway
    benchmark = Benchmark(
        train_puzzles=puzzles,
        test_puzzles=puzzles,
    )

    env = SingleTaskPushWorldEnvironment()
    env_params = env.default_params()
    env_params = env_params.replace(benchmark=benchmark)
    env = GoalObservationWrapper(env)

    params = train_info["runner_state"].params
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
    total_reward = 0
    rendered_imgs = []

    rng = jax.random.key(eval_seed)
    rng, _rng = jax.random.split(rng)

    # initial inputs
    hidden = model.initialize_carry(1)
    prev_reward = jnp.asarray(0)
    prev_action = jnp.asarray(0)

    timestep = reset_fn(env_params, _rng)
    rendered_imgs.append(text_to_rgb(timestep.state.goal_pos, timestep.observation["img"].squeeze(-1)))

    while not timestep.last():
        rng, _rng = jax.random.split(rng)
        dist, _, hidden = apply_fn(
            params,
            {
                "obs_img": timestep.observation["img"][None, None, ...],
                "obs_goal": timestep.observation["goal"][None, None, ...],
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
        rendered_imgs.append(text_to_rgb(timestep.state.goal_pos, timestep.observation["img"].squeeze(-1)))

    imageio.mimsave(f"{video_name}.mp4", rendered_imgs, fps=16, format="mp4")
    return total_reward
