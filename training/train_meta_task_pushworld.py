# Adapted from PureJaxRL implementation and minigrid baselines, source:
# https://github.com/lupuandr/explainable-policies/blob/50acbd777dc7c6d6b8b7255cd1249e81715bcb54/purejaxrl/ppo_rnn.py#L4
# https://github.com/lcswillems/rl-starter-files/blob/master/model.py
import os
import shutil
import time
from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import orbax
import pyrallis
import wandb
from flax.jax_utils import replicate, unreplicate
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from nn_pushworld import ActorCriticRNN
from utils_pushworld import (
    MetaRolloutStats,
    Transition,
    calculate_gae,
    meta_rollout,
    ppo_update_networks,
    rollout,
)

import xminigrid.envs.pushworld as pushworld
from xminigrid.envs.pushworld.benchmarks import Benchmark
from xminigrid.envs.pushworld.environment import Environment, EnvParams, EnvParamsT
from xminigrid.envs.pushworld.envs.meta_task_pushworld import MetaTaskPushWorldEnvironment
from xminigrid.envs.pushworld.wrappers import GoalObservationWrapper, GymAutoResetWrapper

# this will be default in new jax versions anyway
jax.config.update("jax_threefry_partitionable", True)


@dataclass
class TrainConfig:
    project: str = "PushWorld"
    group: str = "default"
    name: str = "meta-task-ppo"
    benchmark_id: str = "level0_mini"
    # If True, track the training progress to wandb
    track: bool = False
    checkpoint_path: Optional[str] = None
    # Upload to W&B
    upload_model: bool = False

    train_test_same: bool = False
    num_train: Optional[int] = None
    num_test: Optional[int] = None

    img_obs: bool = False
    # agent
    obs_emb_dim: int = 16
    action_emb_dim: int = 16
    rnn_hidden_dim: int = 1024
    rnn_num_layers: int = 1
    head_hidden_dim: int = 256
    # training
    enable_bf16: bool = False
    num_envs: int = 8192
    num_steps_per_env: int = 4096
    num_steps_per_update: int = 32
    update_epochs: int = 1
    num_minibatches: int = 16
    total_timesteps: int = 100_000_000
    lr: float = 0.001
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    eval_num_envs: int = 512
    eval_num_episodes: int = 10
    eval_seed: int = 42
    train_seed: int = 42
    # Seed for sampling train and test puzzles if are taking a subset of the benchmark puzzles
    puzzle_seed: int = 42
    checkpoint_path: Optional[str] = None

    def __post_init__(self):
        num_devices = jax.local_device_count()
        # splitting computation across all available devices
        self.num_envs_per_device = self.num_envs // num_devices
        self.total_timesteps_per_device = self.total_timesteps // num_devices
        self.eval_num_envs_per_device = self.eval_num_envs // num_devices
        assert self.num_envs % num_devices == 0
        self.num_meta_updates = round(
            self.total_timesteps_per_device / (self.num_envs_per_device * self.num_steps_per_env)
        )
        self.num_inner_updates = self.num_steps_per_env // self.num_steps_per_update
        assert self.num_steps_per_env % self.num_steps_per_update == 0
        print(f"Num devices: {num_devices}, Num meta updates: {self.num_meta_updates}")


def make_states(config: TrainConfig):
    # for learning rage scheduling
    def linear_schedule(count):
        total_inner_updates = config.num_minibatches * config.update_epochs * config.num_inner_updates
        frac = 1.0 - (count // total_inner_updates) / config.num_meta_updates
        return config.lr * frac

    # setup environment
    # if "XLand" not in config.env_id:
    #     raise ValueError("Only meta-task environments are supported.")

    env = MetaTaskPushWorldEnvironment()
    env_params = env.default_params()
    env = GymAutoResetWrapper(env)
    env = GoalObservationWrapper(env)

    # enabling image observations if needed
    # if config.img_obs:
    #   from xminigrid.experimental.img_obs import RGBImgObservationWrapper
    # Need to add a PushWorld version for this
    #   env = RGBImgObservationWrapper(env)

    # loading benchmark
    benchmark = pushworld.load_benchmark(config.benchmark_id)

    puzzle_rng = jax.random.key(config.puzzle_seed)
    train_rng, test_rng = jax.random.split(puzzle_rng)

    if config.num_train is not None:
        assert (
            config.num_train <= benchmark.num_train_puzzles()
        ), "num_train is larger than num train available in benchmark"
        perm = jax.random.permutation(train_rng, benchmark.num_train_puzzles())
        idxs = perm[: config.num_train]
        benchmark = benchmark.replace(train_puzzles=benchmark.train_puzzles[idxs])
    else:
        config.num_train = benchmark.num_train_puzzles()

    if config.num_test is not None:
        assert (
            config.num_test <= benchmark.num_test_puzzles()
        ), "num_test is larger than num test available in benchmark"
        perm = jax.random.permutation(test_rng, benchmark.num_test_puzzles())
        idxs = perm[: config.num_test]
        benchmark = benchmark.replace(test_puzzles=benchmark.test_puzzles[idxs])
    else:
        config.num_test = benchmark.num_test_puzzles()

    if config.train_test_same:
        benchmark = benchmark.replace(test_puzzles=benchmark.train_puzzles)
        config.num_test = config.num_train

    # set up training state
    rng = jax.random.key(config.train_seed)
    rng, _rng = jax.random.split(rng)

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
    # [batch_size, seq_len, ...]
    shapes = env.observation_shape(env_params)

    init_obs = {
        # We add single channel dimension to end of obs_img
        "obs_img": jnp.zeros((config.num_envs_per_device, 1, *shapes["img"])),
        "obs_goal": jnp.zeros((config.num_envs_per_device, 1, shapes["goal"])),
        "prev_action": jnp.zeros((config.num_envs_per_device, 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((config.num_envs_per_device, 1)),
    }
    init_hstate = network.initialize_carry(batch_size=config.num_envs_per_device)

    network_params = network.init(_rng, init_obs, init_hstate)
    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.inject_hyperparams(optax.adam)(learning_rate=linear_schedule, eps=1e-8),  # eps=1e-5
    )
    train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

    return rng, env, env_params, benchmark, init_hstate, train_state


def make_train(
    env: Environment,
    env_params: EnvParams,
    benchmark: Benchmark,
    config: TrainConfig,
):
    @partial(jax.pmap, axis_name="devices")
    def train(
        rng: jax.Array,
        train_state: TrainState,
        init_hstate: jax.Array,
    ):
        eval_hstate = init_hstate[0][None]

        # META TRAIN LOOP
        def _meta_step(meta_state, _):
            rng, train_state = meta_state

            # INIT ENV
            rng, _rng1, _rng2 = jax.random.split(rng, num=3)
            puzzle_rng = jax.random.split(_rng1, num=config.num_envs_per_device)
            reset_rng = jax.random.split(_rng2, num=config.num_envs_per_device)

            # sample puzzles for this meta update
            puzzles = jax.vmap(benchmark.sample_puzzle, in_axes=(0, None))(puzzle_rng, "train")
            meta_env_params = env_params.replace(puzzle=puzzles)

            timestep = jax.vmap(env.reset, in_axes=(0, 0))(meta_env_params, reset_rng)
            prev_action = jnp.zeros(config.num_envs_per_device, dtype=jnp.int32)
            prev_reward = jnp.zeros(config.num_envs_per_device)

            # INNER TRAIN LOOP
            def _update_step(runner_state, _):
                # COLLECT TRAJECTORIES
                def _env_step(runner_state, _):
                    rng, train_state, prev_timestep, prev_action, prev_reward, prev_hstate = runner_state

                    # SELECT ACTION
                    rng, _rng = jax.random.split(rng)
                    dist, value, hstate = train_state.apply_fn(
                        train_state.params,
                        {
                            # [batch_size, seq_len=1, ...]
                            # We add single channel dimension to end of obs_img
                            "obs_img": prev_timestep.observation["img"][:, None],
                            "obs_goal": prev_timestep.observation["goal"][:, None],
                            "prev_action": prev_action[:, None],
                            "prev_reward": prev_reward[:, None],
                        },
                        prev_hstate,
                    )
                    action, log_prob = dist.sample_and_log_prob(seed=_rng)
                    # squeeze seq_len where possible
                    action, value, log_prob = action.squeeze(1), value.squeeze(1), log_prob.squeeze(1)

                    # STEP ENV
                    timestep = jax.vmap(env.step, in_axes=0)(meta_env_params, prev_timestep, action)
                    transition = Transition(
                        # ATTENTION: done is always false, as we optimize for entire meta-rollout
                        done=jnp.zeros_like(timestep.last()),
                        action=action,
                        value=value,
                        reward=timestep.reward,
                        log_prob=log_prob,
                        obs=prev_timestep.observation["img"],
                        goal=prev_timestep.observation["goal"],
                        prev_action=prev_action,
                        prev_reward=prev_reward,
                    )
                    runner_state = (rng, train_state, timestep, action, timestep.reward, hstate)
                    return runner_state, transition

                initial_hstate = runner_state[-1]
                # transitions: [seq_len, batch_size, ...]
                runner_state, transitions = jax.lax.scan(_env_step, runner_state, None, config.num_steps_per_update)

                # CALCULATE ADVANTAGE
                rng, train_state, timestep, prev_action, prev_reward, hstate = runner_state
                # calculate value of the last step for bootstrapping
                _, last_val, _ = train_state.apply_fn(
                    train_state.params,
                    {
                        # We add single channel dimension to end of obs_img
                        "obs_img": timestep.observation["img"][:, None],
                        "obs_goal": timestep.observation["goal"][:, None],
                        "prev_action": prev_action[:, None],
                        "prev_reward": prev_reward[:, None],
                    },
                    hstate,
                )
                advantages, targets = calculate_gae(transitions, last_val.squeeze(1), config.gamma, config.gae_lambda)

                # UPDATE NETWORK
                def _update_epoch(update_state, _):
                    def _update_minbatch(train_state, batch_info):
                        init_hstate, transitions, advantages, targets = batch_info
                        new_train_state, update_info = ppo_update_networks(
                            train_state=train_state,
                            transitions=transitions,
                            init_hstate=init_hstate.squeeze(1),
                            advantages=advantages,
                            targets=targets,
                            clip_eps=config.clip_eps,
                            vf_coef=config.vf_coef,
                            ent_coef=config.ent_coef,
                        )
                        return new_train_state, update_info

                    rng, train_state, init_hstate, transitions, advantages, targets = update_state

                    # MINIBATCHES PREPARATION
                    rng, _rng = jax.random.split(rng)
                    permutation = jax.random.permutation(_rng, config.num_envs_per_device)
                    # [seq_len, batch_size, ...]
                    batch = (init_hstate, transitions, advantages, targets)
                    # [batch_size, seq_len, ...], as our model assumes
                    batch = jtu.tree_map(lambda x: x.swapaxes(0, 1), batch)

                    shuffled_batch = jtu.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                    # [num_minibatches, minibatch_size, ...]
                    minibatches = jtu.tree_map(
                        lambda x: jnp.reshape(x, (config.num_minibatches, -1) + x.shape[1:]), shuffled_batch
                    )
                    train_state, update_info = jax.lax.scan(_update_minbatch, train_state, minibatches)

                    update_state = (rng, train_state, init_hstate, transitions, advantages, targets)
                    return update_state, update_info

                # hstate shape: [seq_len=None, batch_size, num_layers, hidden_dim]
                update_state = (rng, train_state, initial_hstate[None, :], transitions, advantages, targets)
                update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)
                # WARN: do not forget to get updated params
                rng, train_state = update_state[:2]

                # averaging over minibatches then over epochs
                loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)
                runner_state = (rng, train_state, timestep, prev_action, prev_reward, hstate)
                return runner_state, loss_info

            # on each meta-update we reset rnn hidden to init_hstate
            runner_state = (rng, train_state, timestep, prev_action, prev_reward, init_hstate)
            runner_state, loss_info = jax.lax.scan(_update_step, runner_state, None, config.num_inner_updates)
            # WARN: do not forget to get updated params
            rng, train_state = runner_state[:2]

            # EVALUATE AGENT
            eval_reset_rng = jax.random.key(config.eval_seed)
            eval_test_rng, eval_train_rng = jax.random.split(eval_reset_rng)
            assert config.num_test is not None, "num_test must be set for evaluation"
            assert config.num_train is not None, "num_train must be set for evaluation"

            # Eval on test set
            # Here we are technically just ignoring the eval_num_envs config. This causes eval_num_envs_per_device to
            # become void as well.
            eval_test_reset_rng = jax.random.split(eval_test_rng, num=config.num_test)
            eval_test_puzzles = benchmark.get_test_puzzles()
            eval_test_stats = jax.vmap(meta_rollout, in_axes=(0, None, None, 0, None, None, None))(
                eval_test_reset_rng,
                env,
                meta_env_params,
                eval_test_puzzles,
                train_state,
                eval_hstate,
                config.eval_num_episodes,
            )
            eval_test_stats = jax.lax.pmean(eval_test_stats, axis_name="devices")

            # Eval on train set
            eval_train_reset_rng = jax.random.split(eval_train_rng, num=config.num_train)
            eval_train_puzzles = benchmark.get_train_puzzles()
            eval_train_stats = jax.vmap(meta_rollout, in_axes=(0, None, None, 0, None, None, None))(
                eval_train_reset_rng,
                env,
                meta_env_params,
                eval_train_puzzles,
                train_state,
                eval_hstate,
                config.eval_num_episodes,
            )
            eval_train_stats = jax.lax.pmean(eval_train_stats, axis_name="devices")

            # averaging over inner updates, adding evaluation metrics
            loss_info = jtu.tree_map(lambda x: x.mean(-1), loss_info)
            loss_info.update(
                {
                    # Originally we divided returns_mean by config.eval_num_episodes, but we realized
                    # that it is more intuitive to think about the cumulative returns over the whole meta-episode.
                    "eval_test/returns_mean": eval_test_stats.total_reward.mean(),
                    "eval_train/returns_mean_train": eval_train_stats.total_reward.mean(),
                    "eval_test/returns_median": jnp.median(eval_test_stats.total_reward),
                    "eval_test/returns_20percentile": jnp.percentile(eval_test_stats.total_reward, q=20),
                    # Our definition of solved is whether the agent solves the last trial in the meta-episode.
                    "eval_test/solved_percentage": eval_test_stats.episode_solved[:, -1].mean(),
                    "eval_train/solved_percentage_train": eval_train_stats.episode_solved[:, -1].mean(),
                    "eval_test/lengths": eval_test_stats.episode_lengths.mean(),
                    "eval_test/lengths_20percentile": jnp.percentile(eval_test_stats.episode_lengths, q=20),
                    "lr": train_state.opt_state[-1].hyperparams["learning_rate"],
                    # Store episode arrays - we'll convert to individual metrics outside JIT
                    "eval_test/episode_rewards": eval_test_stats.episode_rewards.mean(axis=0),
                    "eval_test/episode_solved_rates": eval_test_stats.episode_solved.mean(axis=0),
                    "eval_train/episode_rewards": eval_train_stats.episode_rewards.mean(axis=0),
                    "eval_train/episode_solved_rates": eval_train_stats.episode_solved.mean(axis=0),
                }
            )

            meta_state = (rng, train_state)
            return meta_state, loss_info

        meta_state = (rng, train_state)
        meta_state, loss_info = jax.lax.scan(_meta_step, meta_state, None, config.num_meta_updates)
        return {"state": meta_state[-1], "loss_info": loss_info}

    return train


def train(config: TrainConfig):
    # removing existing checkpoints if any
    if config.checkpoint_path is not None and os.path.exists(config.checkpoint_path):
        shutil.rmtree(config.checkpoint_path)

    rng, env, env_params, benchmark, init_hstate, train_state = make_states(config)
    # replicating args across devices
    rng = jax.random.split(rng, num=jax.local_device_count())
    train_state = replicate(train_state, jax.local_devices())
    init_hstate = replicate(init_hstate, jax.local_devices())

    print("Compiling...")
    t = time.time()
    train_fn = make_train(env, env_params, benchmark, config)
    train_fn = train_fn.lower(rng, train_state, init_hstate).compile()
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s.")

    print("Training...")
    t = time.time()
    train_info = jax.block_until_ready(train_fn(rng, train_state, init_hstate))
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s.")

    return unreplicate(train_info), elapsed_time


def processing(config: TrainConfig, train_info, elapsed_time):
    print("Logginig...")
    loss_info = train_info["loss_info"]

    if config.track or config.upload_model:
        run = wandb.init(
            project=config.project,
            group=config.group,
            name=config.name,
            config=asdict(config),
            save_code=True,
        )

    if config.track:
        total_transitions = 0

        # I want to manipulate episode_rewards and episode_solved_rates so that I generate
        # eval_num_episodes separate metrics, where each metric represents the returns or solved rate
        # for one episode in the meta-episode, over meta_updates.
        for episode_idx in range(config.eval_num_episodes):
            loss_info[f"eval_test/episode_rewards/{episode_idx}"] = loss_info["eval_test/episode_rewards"].swapaxes(
                0, 1
            )[episode_idx]
            loss_info[f"eval_test/episode_solved_rates/{episode_idx}"] = loss_info[
                "eval_test/episode_solved_rates"
            ].swapaxes(0, 1)[episode_idx]
            loss_info[f"eval_train/episode_rewards/{episode_idx}"] = loss_info["eval_train/episode_rewards"].swapaxes(
                0, 1
            )[episode_idx]
            loss_info[f"eval_train/episode_solved_rates/{episode_idx}"] = loss_info[
                "eval_train/episode_solved_rates"
            ].swapaxes(0, 1)[episode_idx]

        # Remove the episode_rewards and episode_solved_rates arrays
        loss_info.pop("eval_test/episode_rewards")
        loss_info.pop("eval_test/episode_solved_rates")
        loss_info.pop("eval_train/episode_rewards")
        loss_info.pop("eval_train/episode_solved_rates")

        for i in range(config.num_meta_updates):
            total_transitions += config.num_steps_per_env * config.num_envs_per_device * jax.local_device_count()
            info = jtu.tree_map(lambda x: x[i].item(), loss_info)
            info["transitions"] = total_transitions
            wandb.log(info)

        run.summary["training_time"] = elapsed_time
        run.summary["steps_per_second"] = (config.total_timesteps_per_device * jax.local_device_count()) / elapsed_time

    if config.checkpoint_path is not None:
        checkpoint = {"config": asdict(config), "params": train_info["state"].params}
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(checkpoint)
        orbax_checkpointer.save(config.checkpoint_path, checkpoint, save_args=save_args)

        if config.upload_model:
            # Upload to W&B as artifact
            artifact = wandb.Artifact(
                name=f"model-checkpoint-{run.id}", type="model", description="Trained model checkpoint"
            )
            artifact.add_dir(config.checkpoint_path)  # Add entire checkpoint directory
            run.log_artifact(artifact)

    if config.track or config.upload_model:
        run.finish()

    print("Final test return: ", float(loss_info["eval_test/returns_mean"][-1]))
    print("Final train return: ", float(loss_info["eval_train/returns_mean_train"][-1]))
    print("Final test solve rate: ", float(loss_info["eval_test/solved_percentage"][-1]))
    print("Final train solve rate: ", float(loss_info["eval_train/solved_percentage_train"][-1]))


@pyrallis.wrap()
def main(config: TrainConfig):
    train_info, elapsed_time = train(config)
    processing(config, train_info, elapsed_time)


if __name__ == "__main__":
    main()
