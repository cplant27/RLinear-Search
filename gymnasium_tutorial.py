#!/usr/bin/env python
"""
Gymnasium Blackjack Q-Learning Tutorial

This script demonstrates how to use Gymnasium's Blackjack environment along with a simple Q-learning agent.
It learns a policy over a number of episodes and prints out final training statistics including total training time.
Additionally, you can print detailed results after each training episode and also test the learned policy
using a greedy (i.e. best-action) strategy.

Environment Details (Blackjack-v1):
  - Observation: A tuple (player's sum, dealer's showing card, usable ace flag).
  - Actions: 0 = Stick, 1 = Hit.
  - Rewards: +1 for a win, -1 for a loss, 0 for a draw.

Hyperparameters (tweak these for different learning behaviors):
  - num_episodes: Number of episodes to train on.
  - alpha: Learning rate.
  - gamma: Discount factor.
  - epsilon: Exploration rate (for the epsilon-greedy policy).
"""

import random
import time

import gymnasium as gym


def run_blackjack_qlearning(
    env_name="Blackjack-v1",
    num_episodes=500,
    alpha=0.1,
    gamma=1.0,
    epsilon=0.1,
    verbose=False,
):
    """
    Runs Q-learning on the specified Gymnasium Blackjack environment.

    Parameters:
        env_name (str): The Gymnasium environment ID.
        num_episodes (int): Total number of episodes for training.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate for epsilon-greedy action selection.
        verbose (bool): If True, prints episode reward after every episode.
    Returns:
        Q (dict): The learned Q-table.
    """
    # Create the Blackjack environment.
    env = gym.make(env_name)

    # Q-value table: keys are (state, action) pairs; initial values are 0.0.
    Q = {}

    def get_Q(state, action):
        """Helper to get Q-value for a given state and action."""
        return Q.get((state, action), 0.0)

    def update_Q(state, action, reward, next_state, done):
        """Update Q-value using the Q-learning update rule."""
        max_next = (
            max(get_Q(next_state, a) for a in range(env.action_space.n))
            if not done
            else 0.0
        )
        Q[(state, action)] = get_Q(state, action) + alpha * (
            reward + gamma * max_next - get_Q(state, action)
        )

    # Statistics for monitoring training progress.
    total_reward = 0
    wins = 0
    losses = 0
    draws = 0

    # Record the start time.
    start_time = time.time()

    # Run training episodes.
    for episode in range(num_episodes):
        state, info = (
            env.reset()
        )  # state is a tuple: (player_sum, dealer_card, usable_ace)
        done = False
        episode_reward = 0

        while not done:
            # Epsilon-greedy action selection.
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = [get_Q(state, a) for a in range(env.action_space.n)]
                action = q_values.index(max(q_values))

            next_state, reward, done, truncated, info = env.step(action)
            update_Q(state, action, reward, next_state, done or truncated)
            state = next_state
            episode_reward += reward

            if done or truncated:
                break

        total_reward += episode_reward

        # In Blackjack, reward of +1 indicates a win, -1 a loss, and 0 a draw.
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        else:
            draws += 1

        if verbose:
            print(f"Episode {episode + 1}: Reward = {episode_reward}")

    # Record the end time.
    end_time = time.time()
    training_time = end_time - start_time

    env.close()

    # Print final training statistics.
    print("\nTraining complete!")
    print(f"Total episodes: {num_episodes}")
    print(f"Win rate: {wins / num_episodes * 100:.2f}%")
    print(f"Loss rate: {losses / num_episodes * 100:.2f}%")
    print(f"Draw rate: {draws / num_episodes * 100:.2f}%")
    print(f"Average reward per episode: {total_reward / num_episodes:.2f}")
    print(f"Total training time: {training_time:.2f} seconds")

    return Q


def run_with_learned_policy(Q, env_name="Blackjack-v1", num_episodes=1, verbose=True):
    """
    Runs episodes using the learned Q-values and a greedy policy.

    Parameters:
        Q (dict): The learned Q-table with keys as (state, action) pairs.
        env_name (str): Gymnasium environment ID.
        num_episodes (int): Number of test episodes to run.
        verbose (bool): If True, prints step-by-step details.
    """
    env = gym.make(env_name)
    for episode in range(num_episodes):
        state, info = env.reset()  # Reset environment; state is a tuple.
        done = False
        total_reward = 0
        if verbose:
            print(f"\nTesting Episode {episode + 1} starting with state: {state}")
        while not done:
            # Greedy action: choose action with highest Q-value.
            q_values = [Q.get((state, a), 0.0) for a in range(env.action_space.n)]
            action = q_values.index(max(q_values))
            next_state, reward, done, truncated, info = env.step(action)
            total_reward += reward
            if verbose:
                print(
                    f"State: {state} | Action: {action} | Reward: {reward} | Next state: {next_state}"
                )
            state = next_state
            if done or truncated:
                if verbose:
                    print(f"Episode finished with total reward: {total_reward}\n")
                break
    env.close()


if __name__ == "__main__":
    # Customize these parameters as desired.
    ENV_NAME = "Blackjack-v1"  # Environment ID
    NUM_EPISODES = 1000000  # Number of training episodes
    ALPHA = 0.1  # Learning rate
    GAMMA = 1.0  # Discount factor
    EPSILON = 0.1  # Exploration rate
    VERBOSE = False  # Print episode result after each training episode
    NUM_TEST_EPISODES = 10  # Number of test episodes to run with the learned policy
    TEST_VERBOSE = True  # Print detailed results during testing

    # Run Q-learning training and display final results.
    Q = run_blackjack_qlearning(
        env_name=ENV_NAME,
        num_episodes=NUM_EPISODES,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON,
        verbose=VERBOSE,
    )

    # Run test episodes with the learned Q-values.
    run_with_learned_policy(
        Q, env_name=ENV_NAME, num_episodes=NUM_TEST_EPISODES, verbose=TEST_VERBOSE
    )
