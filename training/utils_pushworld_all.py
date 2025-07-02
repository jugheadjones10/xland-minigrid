# utilities for PPO training and evaluation
import jax
import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState

from xminigrid.envs.pushworld.constants import SUCCESS_REWARD
from xminigrid.envs.pushworld.environment import Environment, EnvParams
from xminigrid.envs.pushworld.types import PushWorldPuzzle


# Training stuff
class Transition(struct.PyTreeNode):
    done: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    log_prob: jax.Array
    # for obs
    obs: jax.Array
    # for rnn policy
    prev_action: jax.Array
    prev_reward: jax.Array


def calculate_gae(
    transitions: Transition,
    last_val: jax.Array,
    gamma: float,
    gae_lambda: float,
) -> tuple[jax.Array, jax.Array]:
    # single iteration for the loop
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        delta = transition.reward + gamma * next_value * (1 - transition.done) - transition.value
        gae = delta + gamma * gae_lambda * (1 - transition.done) * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        transitions,
        reverse=True,
    )
    # advantages and values (Q)
    return advantages, advantages + transitions.value


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
    (loss, vloss, aloss, entropy, grads) = jax.lax.pmean((loss, vloss, aloss, entropy, grads), axis_name="devices")
    train_state = train_state.apply_gradients(grads=grads)
    update_info = {
        "total_loss": loss,
        "value_loss": vloss,
        "actor_loss": aloss,
        "entropy": entropy,
    }
    return train_state, update_info


# for evaluation (evaluate for N consecutive episodes, sum rewards)
# N=1 single task, N>1 for meta-RL
class RolloutStats(struct.PyTreeNode):
    reward: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0.0))
    length: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0))
    episodes: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0))
    solved: jax.Array = struct.field(default_factory=lambda: jnp.asarray(0))


# for tracking per-episode statistics during meta-RL evaluation
class MetaRolloutStats(struct.PyTreeNode):
    episode_rewards: jax.Array  # Shape: [max_episodes] - reward for each episode
    episode_lengths: jax.Array  # Shape: [max_episodes] - length of each episode
    episode_solved: jax.Array  # Shape: [max_episodes] - whether each episode was solved
    total_reward: jax.Array  # Scalar - total reward across all episodes
    num_episodes_completed: jax.Array  # Scalar - number of episodes actually completed


def rollout(
    rng: jax.Array,
    env: Environment,
    env_params: EnvParams,
    eval_puzzle: PushWorldPuzzle,
    train_state: TrainState,
    init_hstate: jax.Array,
    num_consecutive_episodes: int = 1,
) -> RolloutStats:
    def _cond_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, hstate = carry
        return jnp.less(stats.episodes, num_consecutive_episodes)

    def _body_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, hstate = carry

        rng, _rng = jax.random.split(rng)
        dist, _, hstate = train_state.apply_fn(
            train_state.params,
            {
                # We add single channel dimension to end of obs
                "obs": timestep.observation[None, None, ...],
                "prev_action": prev_action[None, None, ...],
                "prev_reward": prev_reward[None, None, ...],
            },
            hstate,
        )
        action = dist.sample(seed=_rng).squeeze()
        timestep = env.step(env_params, timestep, action)

        solved_flag = ((timestep.reward == SUCCESS_REWARD) & (timestep.last() == 1)).astype(jnp.int32)
        stats = stats.replace(
            reward=stats.reward + timestep.reward,
            length=stats.length + 1,
            episodes=stats.episodes + timestep.last(),
            solved=solved_flag,
        )
        carry = (rng, stats, timestep, action, timestep.reward, hstate)
        return carry

    env_params = env_params.replace(puzzle=eval_puzzle)
    timestep = env.eval_reset(env_params, rng)
    prev_action = jnp.asarray(0)
    prev_reward = jnp.asarray(0)
    init_carry = (rng, RolloutStats(), timestep, prev_action, prev_reward, init_hstate)

    final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    return final_carry[1]


def meta_rollout(
    rng: jax.Array,
    env: Environment,
    env_params: EnvParams,
    eval_puzzle: PushWorldPuzzle,
    train_state: TrainState,
    init_hstate: jax.Array,
    num_consecutive_episodes: int = 1,
) -> MetaRolloutStats:
    """Rollout that tracks statistics for each individual episode."""

    def _cond_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, hstate, current_episode_reward, current_episode_length = carry
        return jnp.less(stats.num_episodes_completed, num_consecutive_episodes)

    def _body_fn(carry):
        rng, stats, timestep, prev_action, prev_reward, hstate, current_episode_reward, current_episode_length = carry

        rng, _rng = jax.random.split(rng)
        dist, _, hstate = train_state.apply_fn(
            train_state.params,
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
        )

        # Reset episode accumulators when episode ends
        current_episode_reward = jnp.where(episode_ended, 0.0, current_episode_reward)
        current_episode_length = jnp.where(episode_ended, 0, current_episode_length)

        carry = (rng, stats, timestep, action, timestep.reward, hstate, current_episode_reward, current_episode_length)
        return carry

    # Initialize episode tracking arrays
    episode_rewards = jnp.zeros(num_consecutive_episodes)
    episode_lengths = jnp.zeros(num_consecutive_episodes, dtype=jnp.int32)
    episode_solved = jnp.zeros(num_consecutive_episodes, dtype=jnp.int32)

    initial_stats = MetaRolloutStats(
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        episode_solved=episode_solved,
        total_reward=jnp.asarray(0.0),
        num_episodes_completed=jnp.asarray(0),
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
    )

    final_carry = jax.lax.while_loop(_cond_fn, _body_fn, init_val=init_carry)
    return final_carry[1]
