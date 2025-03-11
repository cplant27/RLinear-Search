import random
from typing import Tuple


class LinearSearchEnv:
    def __init__(self, lower_bound: int = -100, upper_bound: int = 100):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.position = 0
        self.target = None

    def reset(self) -> int:
        """
        Resets the environment:
          - Agent always starts at position 0.
          - Target is randomly chosen within [lower_bound, upper_bound].
        Returns the starting position.
        """
        self.position = 0
        self.target = random.randint(self.lower_bound, self.upper_bound)
        return self.position

    def step(self, action: int) -> Tuple[int, int, bool]:
        """
        Take an action in the environment.
        Actions:
          0 - Move left
          1 - Move right
          2 - Search at current position
        Returns:
          next_state: Updated position.
          reward: Reward for the action.
          done: True if target is found (when searching), False otherwise.
        """
        if action not in [0, 1, 2]:
            raise ValueError(f"Invalid action: {action}. Valid actions are 0, 1, 2.")

        if action == 0:  # Move left
            self.position = max(self.position - 1, self.lower_bound)
            reward = -1
        elif action == 1:  # Move right
            self.position = min(self.position + 1, self.upper_bound)
            reward = -1
        elif action == 2:  # Search action
            reward = 100 if self.position == self.target else -10

        done = action == 2 and self.position == self.target
        return self.position, reward, done
