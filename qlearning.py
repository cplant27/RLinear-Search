import random
from typing import Dict, List
from environment import LinearSearchEnv


def initialize_q_table(
    states: List[int], actions: List[int]
) -> Dict[int, Dict[int, float]]:
    """Initialize the Q-table as a nested dictionary with all zeros."""
    return {state: {action: 0.0 for action in actions} for state in states}


def choose_action(
    state: int, Q: Dict[int, Dict[int, float]], actions: List[int], epsilon: float
) -> int:
    """Choose an action using an epsilon-greedy strategy."""
    if random.random() < epsilon:
        return random.choice(actions)
    return max(Q[state], key=Q[state].get)


def train_q_learning(
    env: LinearSearchEnv,
    Q: Dict[int, Dict[int, float]],
    states: List[int],
    actions: List[int],
    alpha: float,
    gamma: float,
    epsilon: float,
    num_episodes: int,
) -> None:
    """
    Train the Q-learning agent over a number of episodes.
    """
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = choose_action(state, Q, actions, epsilon)
            next_state, reward, done = env.step(action)
            best_next = max(Q[next_state].values())
            Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])
            state = next_state
