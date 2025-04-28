# Infinite Linear Search and Rescue Environment

This project implements a custom reinforcement learning environment using the Gymnasium library. The environment, `InfiniteLinearSearchEnv`, simulates a search and rescue task on a semi-infinite line starting at position 0 and extending infinitely in the positive direction.

## Environment Description

The agent starts at the base (position 0) and must explore the line to find one or more targets located at unknown positions. Once a target is found (by moving within a close proximity), the agent needs to return to the base with the target(s) to complete the rescue. The targets can be configured to be stationary or mobile, following a predefined zigzag pattern.

## Key Features

- **Semi-Infinite Space**: The environment exists on a line `[0, +infinity)`.
- **Search and Rescue Task**: The agent needs to find targets and return them to the base at position 0.
- **Moving Targets**: Optional target movement with configurable probability and speed, following a zigzag pattern.
- **Sensing Capability**: The agent can sense the distance and direction to the nearest unfound target within a specified range.
- **Configurable Parameters**: Environment settings like the number of targets, target range, sensing range, and movement parameters can be customized.
- **Gymnasium Interface**: Follows the standard Gymnasium API for easy integration with RL algorithms.

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    Install the required packages using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    If `requirements.txt` is not available, you can install the primary dependencies directly:
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
    num_targets=2,
    target_range=500,
    move_target=True,
    target_move_prob=0.05,
    sensing_range=50
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
        print(f"Episode finished - Terminated: Target(s) rescued!")
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
- `target_range` (int): Maximum initial placement position for targets.
- `region_size` (int): Size of regions for tracking visitation (used internally).
- `move_target` (bool): Whether targets move during the episode.
- `target_move_prob` (float): Probability of a target moving in a given step.
- `target_speed` (int): Maximum distance a target can move in one step (used within the zigzag pattern).
- `num_targets` (int): Number of targets in the environment.
- `sensing_range` (int): The distance within which the agent can sense nearby targets.
- `seed` (Optional[int]): Seed for the random number generator.

## Observation Space

The observation is a dictionary containing:

- `relative_position`: Agent's position relative to the nearest unfound target (during search) or the base (during rescue).
- `direction`: Agent's current facing direction (0: left, 1: right).
- `farthest_right_rel`: Farthest position reached relative to the current position.
- `target_found`: Binary flag indicating if _any_ target has been found.
- `distance_to_base`: Absolute distance from the agent to the base (position 0).
- `search_phase`: Current phase of the search/rescue process (Internal state, not directly used in default rewards).
- `sensing_info`: A tuple `(distance, direction)` to the nearest unfound target within `sensing_range`. `distance` is -1 if no target is sensed. `direction` is -1 (left), 1 (right), or 0.
- `targets_found_count`: Number of targets currently found.

## Action Space

The action space is discrete with two possible actions:

- `0`: Move left.
- `1`: Move right.

The agent cannot move to a position less than 0.

## Reward Structure

The reward system is designed to guide the agent towards finding and rescuing targets:

- **Large Positive Rewards:**
  - Finding a target (`+200.0`).
  - Rescuing found target(s) by returning to the base (`+500.0` per target rescued in that step).
  - Rescuing _all_ targets (`+1000.0` bonus upon final rescue).
- **Smaller Positive Rewards (Guidance):**
  - Moving right during exploration when unfound targets exist (`+1.0`).
  - Moving left towards the base when carrying targets and all targets are found (`+1.0`).
  - Moving towards a sensed target (`+1.0`).
  - Sustained rightward movement during exploration (momentum bonus).
- **Small Negative Reward:**
  - Living penalty per step (`-0.5`) to encourage efficiency.

## Configuration

The verbosity of the environment's console output can be configured by modifying the constants at the top of `src/environment.py`:

- `PRINT_FREQUENCY`: Controls how often informational messages are printed.
- `PRINT_EXPLORATION`: Toggles exploration-related messages.
- `PRINT_REWARDS`: Toggles detailed reward breakdown messages.
- `PRINT_DECISIONS`: Toggles decision-logic related messages (currently unused in default configuration).
