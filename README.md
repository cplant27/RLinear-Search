# Infinite Linear Search and Rescue Environment (Single Target)

This project implements a custom reinforcement learning environment using the Gymnasium library. The environment, `InfiniteLinearSearchEnv`, simulates a search and rescue task for a single target on a semi-infinite line starting at position 0 and extending infinitely in the positive direction.

## Environment Description

The agent starts at a position near the base (default: between 80-120) and must explore the line to find a single target located at an unknown position. Once the target is found (by moving within close proximity), the agent needs to return to the base (position 0) with the target to complete the rescue. The target can be configured to be stationary or mobile, following a predefined zigzag pattern.

## Key Features

- **Semi-Infinite Space**: The environment exists on a line `[0, +infinity)`.
- **Single Target Search and Rescue**: The agent needs to find one target and return it to the base at position 0.
- **Moving Target**: Optional target movement with configurable probability and speed, following a zigzag pattern.
- **Search Phases**: The agent follows a structured search pattern: initial exploration right, return to base, then full exploration right.
- **Configurable Parameters**: Environment settings like target range, movement parameters, and max steps can be customized.
- **Gymnasium Interface**: Follows the standard Gymnasium API for easy integration with RL algorithms.

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Install dependencies:**
    The primary dependencies are Gymnasium and NumPy.
    ```bash
    pip install gymnasium numpy
    ```

## How to Use

Here's a basic example of how to create and interact with the environment:

```python
import gymnasium as gym
from src.environment import InfiniteLinearSearchEnv  # Assuming your environment file is here

# Create the environment instance
env = InfiniteLinearSearchEnv(
    max_steps=2000,
    target_range=500,
    move_target=True,
    target_move_prob=0.05,
)

# Reset the environment
observation, info = env.reset()

terminated = False
truncated = False
total_reward = 0
step_count = 0

while not terminated and not truncated:
    # Choose an action (0: left, 1: right) - e.g., random action
    action = env.action_space.sample()

    # Take a step
    observation, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    step_count += 1

    # Optional: Render the environment state
    # env.render()

    if terminated:
        print(f"Episode finished - Terminated: Target rescued!")
    if truncated:
        print(f"Episode finished - Truncated: Max steps reached.")

print(f"Total steps: {step_count}")
print(f"Total reward: {total_reward}")
print(f"Final Info: {info}")

env.close()
```

## Environment Parameters

The `InfiniteLinearSearchEnv` class accepts the following parameters during initialization:

- `max_steps` (int): Maximum number of steps per episode.
- `target_range` (int): Maximum initial placement position for the target.
- `region_size` (int): Size of regions for tracking visitation (used internally).
- `move_target` (bool): Whether the target moves during the episode.
- `target_move_prob` (float): Probability of the target moving in a given step.
- `target_speed` (int): Maximum distance the target can move in one step (used within the zigzag pattern).
- `seed` (Optional[int]): Seed for the random number generator.

## Observation Space

The observation is a dictionary containing:

- `relative_position`: Agent's position relative to the target (during search) or the base (during rescue).
- `direction`: Agent's current facing direction (0: left, 1: right).
- `farthest_right_rel`: Farthest position reached relative to the current position.
- `target_found`: Binary flag indicating if the target has been found.
- `distance_to_base`: Absolute distance from the agent to the base (position 0).
- `search_phase`: Current phase of the search (0: initial, 1: return, 2: explore).

## Action Space

The action space is discrete with two possible actions:

- `0`: Move left.
- `1`: Move right.

The agent cannot move to a position less than 0.

## Reward Structure

The reward system guides the agent through search and rescue phases:

- **Base Penalty:** A small negative reward (`-0.1`) is given each step to encourage efficiency.
- **Search Phase Rewards:**
  - **Initial Exploration (Phase 0):** Reward for moving right (`+0.3`). Phase ends when agent reaches `initial_exploration_threshold`.
  - **Return to Base (Phase 1):** Reward for moving left towards base (`+0.4`). Phase ends when agent reaches base (position 0).
  - **Full Exploration (Phase 2):** Reward for moving right (`+0.3`) and reaching new farthest right positions (`+0.4`).
  - **Target Found:** A large reward (`+50.0`) is given when the agent is within 1 step of the target. This transitions the agent to the Rescue Phase.
  - **Region Bonus:** Small reward (`+0.2`) for visiting new regions frequently (all search phases).
- **Rescue Phase Rewards (Target Found):**
  - **Rescue Complete:** A very large reward (`+100.0`) is given when the agent returns to the base (position 0) with the target. This terminates the episode.
  - **Base Progress:** Reward for moving left towards the base (`+0.5 * (1 / max(1, current_position / 10))`).
  - **Away Penalty:** Penalty for moving right (away from base) (`-0.3`).
  - **Proximity Bonus:** Small reward for reducing distance to base (`+0.2 * step_size / previous_distance`).

## Configuration

The verbosity of the environment's console output can be configured by modifying the constants at the top of `src/environment.py`:

- `PRINT_FREQUENCY`: Controls how often informational messages are printed.
- `PRINT_EXPLORATION`: Toggles exploration-related messages.
- `PRINT_REWARDS`: Toggles detailed reward breakdown messages.
- `PRINT_DECISIONS`: Toggles decision-logic related messages (currently unused in default configuration).
