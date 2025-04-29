import random
from typing import Any, Dict, List, Tuple

from src.environment import InfiniteLinearSearchEnv


def initialize_q_table(
    env: InfiniteLinearSearchEnv, actions: List[int]
) -> Dict[Tuple, Dict[int, float]]:
    """
    Initialize the Q-table with a default dictionary to handle the large state space.

    With relative position and potentially infinite space, we use a default dictionary approach.

    Args:
        env: The environment instance
        actions: List of possible actions

    Returns:
        A Q-table implemented as a default dictionary
    """
    from collections import defaultdict

    # Default factory creates a new actions dictionary for any new state
    def default_q_value():
        return {action: 0.0 for action in actions}

    # Create default dictionary for Q-table
    q_table = defaultdict(default_q_value)

    # Pre-populate with some common states
    # This is mostly for the starting state
    obs, _ = env.reset()
    initial_state = observation_to_state(obs)
    q_table[initial_state] = {action: 0.0 for action in actions}

    return q_table


def observation_to_state(observation: Dict[str, Any]) -> Tuple:
    """
    Convert an observation dict to a state tuple for the Q-table.
    For infinite line, we need to quantize continuous values to create a finite state space.

    Args:
        observation: Observation dict from the environment

    Returns:
        A tuple representing the state with quantized values
    """
    # Extract values from observation
    # Note: environment now returns correct types, no need for np.array indexing [0]
    direction = observation["direction"]
    target_found = observation["target_found"]
    distance_to_base = float(
        observation["distance_to_base"][0]
    )  # Still need float from array
    search_phase = observation["search_phase"]
    farthest_right_rel = float(
        observation["farthest_right_rel"][0]
    )  # Use correct key, need float from array

    # Quantize distance to base
    if distance_to_base < 20:
        quantized_base_distance = int(distance_to_base)
    elif distance_to_base < 100:
        quantized_base_distance = int(distance_to_base / 5) * 5
    else:
        quantized_base_distance = int(distance_to_base / 20) * 20

    # Quantize farthest right relative position
    quantized_farthest_right_rel = int(farthest_right_rel / 50) * 50

    # Create a hashable state tuple (removed quantized_position)
    return (
        direction,
        quantized_farthest_right_rel,
        target_found,
        quantized_base_distance,
        search_phase,
    )


def choose_action(
    state: Tuple, Q: Dict[Tuple, Dict[int, float]], actions: List[int], epsilon: float
) -> int:
    """
    Choose an action using epsilon-greedy strategy with adaptive exploration bias.

    Args:
        state: Current state tuple
        Q: Q-table
        actions: Available actions
        epsilon: Exploration rate

    Returns:
        Selected action
    """
    # Extract information from state
    (
        direction,
        farthest_right,
        target_found,
        base_distance,
        search_phase,
    ) = state

    if random.random() < epsilon:
        # Exploration phase - with some intelligent bias based on search phase

        # If target found, bias toward returning to base
        if target_found == 1:
            # Strong bias to move left toward base
            if base_distance > 0 and random.random() < 0.85:
                return 0  # Go left toward base
            else:
                return random.choice(actions)  # Random exploration

        # If not found, bias based on search phase
        else:
            # Phase 0: Initial exploration - bias toward going right
            if search_phase == 0 and random.random() < 0.8:
                return 1  # Go right

            # Phase 1: Return to base - bias toward going left
            elif search_phase == 1 and random.random() < 0.8:
                return 0  # Go left

            # Phase 2: Full exploration right - bias toward going right
            elif search_phase == 2 and random.random() < 0.8:
                return 1  # Go right

            # Otherwise random choice
            return random.choice(actions)
    else:
        # Exploitation - choose best action from Q-table
        return max(Q[state], key=Q[state].get)


def train_q_learning(
    env: InfiniteLinearSearchEnv,
    Q: Dict[Tuple, Dict[int, float]],
    actions: List[int],
    alpha: float,
    gamma: float,
    epsilon: float,
    num_episodes: int,
) -> None:
    """
    Train the Q-learning agent over multiple episodes.

    Args:
        env: The environment
        Q: Q-table
        actions: Available actions
        alpha: Learning rate
        gamma: Discount factor
        epsilon: Exploration rate
        num_episodes: Number of episodes to train for
    """
    # For tracking performance
    successful_episodes = 0
    completed_steps = 0

    # Experience replay buffer for faster learning
    replay_buffer = []
    replay_size = 1000  # Larger buffer to capture more diverse experiences

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = observation_to_state(obs)
        done = False
        truncated = False
        steps = 0
        episode_reward = 0

        # Episode loop
        while not done and not truncated:
            # Choose action with epsilon-greedy
            action = choose_action(state, Q, actions, epsilon)

            # Take action
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = observation_to_state(next_obs)
            steps += 1
            episode_reward += reward

            # Store experience for replay
            replay_buffer.append((state, action, reward, next_state, done))
            if len(replay_buffer) > replay_size:
                replay_buffer.pop(0)

            # Standard Q-learning update with slightly higher learning rate
            best_next = max(Q[next_state].values())
            Q[state][action] += alpha * (reward + gamma * best_next - Q[state][action])

            # Move to next state
            state = next_state

            # Terminate very long episodes
            if steps >= env.max_steps:
                truncated = True

        # Track progress
        completed_steps += steps
        if done:  # Target found
            successful_episodes += 1

        # Experience replay after each episode for faster learning
        if episode > 0 and len(replay_buffer) > 20:
            # Replay a batch of experiences
            batch_size = min(50, len(replay_buffer))
            samples = random.sample(replay_buffer, batch_size)

            for old_state, old_action, old_reward, old_next_state, old_done in samples:
                if not old_done:
                    old_best_next = max(Q[old_next_state].values())
                    Q[old_state][old_action] += (
                        alpha
                        * 0.7  # Slightly reduced learning rate for replay
                        * (
                            old_reward
                            + gamma * old_best_next
                            - Q[old_state][old_action]
                        )
                    )

        # Print progress occasionally
        if (episode + 1) % 10 == 0:
            avg_steps = completed_steps / (episode + 1)
            success_rate = successful_episodes / (episode + 1) * 100
            print(
                f"Training: {episode+1}/{num_episodes} episodes | "
                f"Avg steps: {avg_steps:.1f} | Success rate: {success_rate:.1f}% | "
                f"Last episode reward: {episode_reward:.1f}"
            )
