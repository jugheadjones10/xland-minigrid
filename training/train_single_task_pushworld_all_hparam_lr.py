# Adapted from PureJaxRL implementation and minigrid baselines, source:
# https://github.com/lupuandr/explainable-policies/blob/50acbd777dc7c6d6b8b7255cd1249e81715bcb54/purejaxrl/ppo_rnn.py#L4
# https://github.com/lcswillems/rl-starter-files/blob/master/model.py
import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import List, Optional

import imageio
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import orbax
import pyrallis
import wandb
from flax.jax_utils import replicate, unreplicate
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from nn_pushworld_all import ActorCriticRNN
from utils_pushworld_all import Transition, calculate_gae, rollout

import xminigrid.envs.pushworld as pushworld
from xminigrid.envs.pushworld.benchmarks import BenchmarkAll
from xminigrid.envs.pushworld.environment import Environment, EnvParams, EnvParamsT
from xminigrid.envs.pushworld.envs.single_task_all_pushworld import SingleTaskPushWorldEnvironmentAll
from xminigrid.envs.pushworld.wrappers import GymAutoResetWrapper

# this will be default in new jax versions anyway
# jax.config.update("jax_threefry_partitionable", True)


def ppo_update_networks(
    train_state: TrainState,
    transitions: Transition,
    init_hstate: jax.Array,
    advantages: jax.Array,
    targets: jax.Array,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
):
    # NORMALIZE ADVANTAGES
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def _loss_fn(params):
        # RERUN NETWORK
        dist, value, _ = train_state.apply_fn(
            params,
            {
                # [batch_size, seq_len, ...]
                "obs": transitions.obs,
                "prev_action": transitions.prev_action,
                "prev_reward": transitions.prev_reward,
            },
            init_hstate,
        )
        log_prob = dist.log_prob(transitions.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = transitions.value + (value - transitions.value).clip(-clip_eps, clip_eps)
        value_loss = jnp.square(value - targets)
        value_loss_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = 0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()

        # TODO: ablate this!
        # value_loss = jnp.square(value - targets).mean()

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - transitions.log_prob)
        actor_loss1 = advantages * ratio
        actor_loss2 = advantages * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()
        entropy = dist.entropy().mean()

        total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy
        return total_loss, (value_loss, actor_loss, entropy)

    (loss, (vloss, aloss, entropy)), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
    train_state = train_state.apply_gradients(grads=grads)
    update_info = {
        "total_loss": loss,
        "value_loss": vloss,
        "actor_loss": aloss,
        "entropy": entropy,
    }
    return train_state, update_info


@dataclass
class TrainConfig:
    project: str = "PushWorld"
    group: str = "default"
    name: str = "single-task-ppo-pushworld"
    benchmark_id: str = "level0_mini"
    # If True, track the training progress to wandb
    track: bool = False
    checkpoint_path: Optional[str] = None
    # Upload to W&B
    upload_model: bool = False

    # If True, test puzzles are duplicated from train puzzles
    train_test_same: bool = False
    num_train: Optional[int] = None
    num_test: Optional[int] = None

    # For lr hparam tuning
    lr_hparams: List[float] = field(default_factory=lambda: [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01])

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
    num_steps: int = 16
    update_epochs: int = 1
    num_minibatches: int = 16
    total_timesteps: int = 1_000_000
    lr: float = 0.001
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    eval_episodes: int = 80
    seed: int = 42
    # Seed for sampling train and test puzzles if are taking a subset of the benchmark puzzles
    puzzle_seed: int = 42
    eval_seed: int = 42

    def __post_init__(self):
        num_devices = jax.local_device_count()
        # splitting computation across all available devices
        self.num_envs_per_device = self.num_envs // num_devices
        self.total_timesteps_per_device = self.total_timesteps // num_devices
        self.eval_episodes_per_device = self.eval_episodes // num_devices
        assert self.num_envs % num_devices == 0
        self.num_updates = self.total_timesteps_per_device // self.num_steps // self.num_envs_per_device
        print(f"Num devices: {num_devices}, Num updates: {self.num_updates}")


def make_states(config: TrainConfig):
    # setup environment
    env = SingleTaskPushWorldEnvironmentAll()
    env_params = env.default_params()
    env = GymAutoResetWrapper(env)

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

    # enabling image observations if needed
    # if config.img_obs:
    #     from xminigrid.experimental.img_obs import RGBImgObservationWrapper
    #     env = RGBImgObservationWrapper(env)

    # setup training state
    rng = jax.random.key(config.seed)
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
        "obs": jnp.zeros((config.num_envs_per_device, 1, *shapes)),
        "prev_action": jnp.zeros((config.num_envs_per_device, 1), dtype=jnp.int32),
        "prev_reward": jnp.zeros((config.num_envs_per_device, 1)),
    }
    init_hstate = network.initialize_carry(batch_size=config.num_envs_per_device)

    network_params = network.init(_rng, init_obs, init_hstate)

    return rng, env, env_params, benchmark, init_hstate, network, network_params


def make_train(
    env: Environment,
    env_params: EnvParams,
    benchmark: BenchmarkAll,
    config: TrainConfig,
    network: ActorCriticRNN,
    network_params: jax.Array,
):
    def train(
        lr: jax.Array,
        rng: jax.Array,
        init_hstate: jax.Array,
    ):
        def linear_schedule(count):
            frac = 1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates
            return lr * frac

        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(learning_rate=linear_schedule, eps=1e-8),  # eps=1e-5
        )
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs_per_device)

        puzzle_env_params = env_params.replace(benchmark=benchmark)

        timestep = jax.vmap(env.reset, in_axes=(None, 0))(puzzle_env_params, reset_rng)
        prev_action = jnp.zeros(config.num_envs_per_device, dtype=jnp.int32)
        prev_reward = jnp.zeros(config.num_envs_per_device)

        # TRAIN LOOP
        def _update_step(runner_state, update_idx):
            # jax.debug.print("Update step: {}", update_idx)

            # COLLECT TRAJECTORIES
            def _env_step(runner_state, _):
                rng, train_state, prev_timestep, prev_action, prev_reward, prev_hstate = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                dist, value, hstate = train_state.apply_fn(
                    train_state.params,
                    {
                        # [batch_size, seq_len=1, ...]
                        "obs": prev_timestep.observation[:, None],
                        "prev_action": prev_action[:, None],
                        "prev_reward": prev_reward[:, None],
                    },
                    prev_hstate,
                )
                action, log_prob = dist.sample_and_log_prob(seed=_rng)
                # squeeze seq_len where possible
                action, value, log_prob = action.squeeze(1), value.squeeze(1), log_prob.squeeze(1)

                # STEP ENV
                timestep = jax.vmap(env.step, in_axes=(None, 0, 0))(puzzle_env_params, prev_timestep, action)
                transition = Transition(
                    done=timestep.last(),
                    action=action,
                    value=value,
                    reward=timestep.reward,
                    log_prob=log_prob,
                    obs=prev_timestep.observation,
                    prev_action=prev_action,
                    prev_reward=prev_reward,
                )
                runner_state = (rng, train_state, timestep, action, timestep.reward, hstate)
                return runner_state, transition

            initial_hstate = runner_state[-1]
            # transitions: [seq_len, batch_size, ...]
            runner_state, transitions = jax.lax.scan(_env_step, runner_state, None, config.num_steps)

            # CALCULATE ADVANTAGE
            rng, train_state, timestep, prev_action, prev_reward, hstate = runner_state
            # calculate value of the last step for bootstrapping
            _, last_val, _ = train_state.apply_fn(
                train_state.params,
                {
                    "obs": timestep.observation[:, None],
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

            # [seq_len, batch_size, num_layers, hidden_dim]
            init_hstate = initial_hstate[None, :]
            update_state = (rng, train_state, init_hstate, transitions, advantages, targets)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)

            # averaging over minibatches then over epochs
            loss_info = jtu.tree_map(lambda x: x.mean(-1).mean(-1), loss_info)

            rng, train_state = update_state[:2]

            # EVALUATE AGENT
            eval_reset_rng = jax.random.key(config.eval_seed)
            eval_test_rng, eval_train_rng = jax.random.split(eval_reset_rng)
            assert config.num_test is not None, "num_test must be set for evaluation"
            assert config.num_train is not None, "num_train must be set for evaluation"

            eval_test_reset_rng = jax.random.split(eval_test_rng, num=config.num_test)
            eval_test_puzzles = benchmark.get_test_puzzles()
            # vmap only on rngs
            eval_test_stats = jax.vmap(rollout, in_axes=(0, None, None, 0, None, None, None))(
                eval_test_reset_rng,
                env,
                puzzle_env_params,
                eval_test_puzzles,
                train_state,
                # TODO: make this as a static method mb?
                # jnp.zeros((1, config.rnn_num_layers, config.rnn_hidden_dim)),
                network.initialize_carry(batch_size=config.num_envs_per_device),
                1,
            )

            # Eval on train set
            eval_train_reset_rng = jax.random.split(eval_train_rng, num=config.num_train)
            eval_train_puzzles = benchmark.get_train_puzzles()
            eval_train_stats = jax.vmap(rollout, in_axes=(0, None, None, 0, None, None, None))(
                eval_train_reset_rng,
                env,
                puzzle_env_params,
                eval_train_puzzles,
                train_state,
                # TODO: make this as a static method mb?
                # jnp.zeros((1, config.rnn_num_layers, config.rnn_hidden_dim)),
                network.initialize_carry(batch_size=config.num_envs_per_device),
                1,
            )

            loss_info.update(
                {
                    "eval_test/returns_mean": eval_test_stats.reward.mean(0),
                    "eval_test/lengths": eval_test_stats.length.mean(0),
                    "eval_test/solved_percentage": eval_test_stats.solved.sum(0) / config.num_test,
                    "eval_train/returns_mean": eval_train_stats.reward.mean(0),
                    "eval_train/lengths": eval_train_stats.length.mean(0),
                    "eval_train/solved_percentage": eval_train_stats.solved.sum(0) / config.num_train,
                    "lr": train_state.opt_state[-1].hyperparams["learning_rate"],
                }
            )
            runner_state = (rng, train_state, timestep, prev_action, prev_reward, hstate)
            return runner_state, loss_info

        runner_state = (rng, train_state, timestep, prev_action, prev_reward, init_hstate)
        # Create a sequence of numbers from 0 to num_updates-1 for progress tracking
        update_indices = jnp.arange(config.num_updates)
        runner_state, loss_info = jax.lax.scan(_update_step, runner_state, update_indices, config.num_updates)
        # return {"runner_state": runner_state[1], "loss_info": loss_info}
        return {"loss_info": loss_info}

    return train


def train(config: TrainConfig):
    # with jax.checking_leaks():
    # removing existing checkpoints if any
    if config.checkpoint_path is not None and os.path.exists(config.checkpoint_path):
        shutil.rmtree(config.checkpoint_path)

    rng, env, env_params, benchmark, init_hstate, network, network_params = make_states(config)

    lr_search = jnp.array(config.lr_hparams)
    train_fn = make_train(env, env_params, benchmark, config, network, network_params)
    train_fn_vmap = jax.jit(jax.vmap(train_fn, in_axes=(0, None, None)))

    print("Compiling...")
    t = time.time()
    train_fn_jit_vmap = train_fn_vmap.lower(lr_search, rng, init_hstate).compile()
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s.")

    print("Training...")
    t = time.time()
    train_info = jax.block_until_ready(train_fn_jit_vmap(lr_search, rng, init_hstate))
    elapsed_time = time.time() - t
    print(f"Done in {elapsed_time:.2f}s.")

    return train_info, elapsed_time


def processing(config: TrainConfig, train_info, elapsed_time):
    print("Logging...")
    loss_info = train_info["loss_info"]

    if config.track:
        for j, lr in enumerate(config.lr_hparams):
            lr_run = wandb.init(
                project=config.project,
                group=config.group,
                name=f"{config.name}|lr={lr}",
                config={**asdict(config), "lr_selected": float(lr)},
                save_code=True,
                reinit=True,
            )

            total_transitions = 0
            for i in range(config.num_updates):
                total_transitions += config.num_steps * config.num_envs_per_device * jax.local_device_count()
                info_ji = jtu.tree_map(lambda x: x[j, i].item(), loss_info)
                info_ji["transitions"] = total_transitions
                wandb.log(info_ji)

            lr_run.summary["training_time"] = elapsed_time
            lr_run.summary["steps_per_second"] = (
                config.total_timesteps_per_device * jax.local_device_count()
            ) / elapsed_time
            lr_run.summary["final/returns_mean"] = float(loss_info["eval_test/returns_mean"][j, -1])
            lr_run.summary["final/solved_pct"] = float(loss_info["eval_test/solved_percentage"][j, -1])
            lr_run.finish()

    # if config.checkpoint_path is not None:
    #     checkpoint = {"config": asdict(config), "params": train_info["runner_state"].params}
    #     orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    #     save_args = orbax_utils.save_args_from_target(checkpoint)
    #     orbax_checkpointer.save(config.checkpoint_path, checkpoint, save_args=save_args)

    #     if config.upload_model:
    #         # Upload to W&B as artifact
    #         artifact = wandb.Artifact(
    #             name=f"model-checkpoint-{run.id}", type="model", description="Trained model checkpoint"
    #         )
    #         artifact.add_dir(config.checkpoint_path)  # Add entire checkpoint directory
    #         run.log_artifact(artifact)

    # if config.track or config.upload_model:
    #     run.finish()

    for i in range(len(config.lr_hparams)):
        print(
            f"Final test set return for lr {config.lr_hparams[i]}: ",
            float(loss_info["eval_test/returns_mean"][i].mean(0)),
        )
        print(
            f"Final test set solved percentage for lr {config.lr_hparams[i]}: ",
            float(loss_info["eval_test/solved_percentage"][i].mean(0)),
        )


@pyrallis.wrap()
def main(config: TrainConfig):
    train_info, elapsed_time = train(config)
    processing(config, train_info, elapsed_time)


if __name__ == "__main__":
    main()
