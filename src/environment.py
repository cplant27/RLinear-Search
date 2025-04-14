import math
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.utils import seeding

# Configure environment verbosity
PRINT_FREQUENCY = 5  # Only print every 5th message to reduce spam
PRINT_EXPLORATION = False  # Print exploration-related messages
PRINT_REWARDS = False  # Print reward-related messages
PRINT_DECISIONS = False  # Print decision-related messages


def print_info(message, category="INFO", step=None, frequency=1):
    """Print formatted information with category and optional step count"""
    if step is not None and step % frequency != 0:
        return  # Skip messages based on frequency

    if category == "EXPLORATION" and not PRINT_EXPLORATION:
        return
    if category == "REWARD" and not PRINT_REWARDS:
        return
    if category == "DECISION" and not PRINT_DECISIONS:
        return

    step_info = f"[Step {step}] " if step is not None else ""
    print(f"[{category}] {step_info}{message}")


def print_summary(message):
    """Print a summary message with highlighting"""
    print("\n" + "=" * 80)
    print(f"SUMMARY: {message}")
    print("=" * 80 + "\n")


class InfiniteLinearSearchEnv(gym.Env):
    """
    Environment for search and rescue on an infinite linear space [0, infinity).

    The agent is placed on an infinite line extending in the positive direction,
    and needs to find a target, then return to the base point (at position 0) on the left side.

    Key features:
    - Infinite line with only left boundary at 0
    - Moving target with configurable movement pattern
    - Search strategy: go right for a bit, then to base, then explore right
    - Rescue: return to base with the target

    Parameters:
        max_steps (int): Maximum number of steps per episode
        target_range (int): Range for initial target placement
        region_size (int): Size of each region for tracking visitation
        move_target (bool): Whether the target should move
        target_move_prob (float): Probability of target moving on each step
        target_speed (int): Maximum distance target can move in one step
    """

    metadata = {"render.modes": ["console"]}

    def __init__(
        self,
        max_steps: int = 1000,
        target_range: int = 500,
        region_size: int = 10,
        move_target: bool = False,
        target_move_prob: float = 0.1,
        target_speed: int = 1,
        seed: Optional[int] = None,
    ):
        super().__init__()

        # Basic environment settings
        self.max_steps = max_steps
        self.target_range = target_range
        self.region_size = region_size

        # Target movement settings
        self.move_target = move_target
        self.target_move_prob = target_move_prob
        self.target_speed = target_speed

        # Zigzag pattern parameters
        self.zigzag_amplitude = 20  # Max distance from the center line
        self.zigzag_period = 30  # Steps to complete one full zigzag cycle
        self.zigzag_phase = 0  # Current phase in the zigzag cycle

        # Base point (rescue location) is at position 0
        self.base_point = 0

        # Search and rescue state
        self.target_found = False
        self.rescue_complete = False

        # Search phases
        self.SEARCH_PHASE_INITIAL = 0  # Initial exploration right
        self.SEARCH_PHASE_RETURN = 1  # Return to base
        self.SEARCH_PHASE_EXPLORE = 2  # Full exploration right
        self.search_phase = self.SEARCH_PHASE_INITIAL

        # Action space: 0 = left, 1 = right
        self.action_space = spaces.Discrete(2)

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "relative_position": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "direction": spaces.Discrete(2),  # 0: left, 1: right
                "farthest_right_rel": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "target_found": spaces.Discrete(2),  # 0: no, 1: yes
                "distance_to_base": spaces.Box(
                    low=0, high=np.inf, shape=(1,), dtype=np.float32
                ),
                "search_phase": spaces.Discrete(3),  # 0: initial, 1: return, 2: explore
            }
        )

        # Initialize state
        self.current_position = 0
        self.target = 0
        self.previous_positions = []
        self.steps_taken = 0
        self.current_direction = 1  # Start facing right
        self.previous_action = None
        self.regions_visited = set()
        self.farthest_right = 0
        self.prev_target_position = 0
        self.target_move_direction = 0

        # Initial exploration threshold
        self.initial_exploration_threshold = 150  # Go right about this far initially

        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        """Set the random seed for the environment."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
            random.seed(seed)

        # Reset agent to starting position (around 100)
        self.current_position = self.np_random.integers(80, 120)
        self.steps_taken = 0
        self.previous_action = None

        # Reset search and rescue state
        self.target_found = False
        self.rescue_complete = False
        self.search_phase = self.SEARCH_PHASE_INITIAL

        # Place target (randomly between 0 and target_range)
        self.target = self.np_random.integers(0, self.target_range)

        # Reset tracking variables
        self.regions_visited = set()
        self.farthest_right = self.current_position
        self.prev_target_position = self.target
        self.target_move_direction = 0

        # Reset zigzag parameters
        self.zigzag_phase = 0
        self.zigzag_center = self.target  # Center line for zigzag pattern

        # Reset positions history
        self.previous_positions = []

        # Record initial region
        self.add_current_region_to_visited()

        observation = self._get_observation()
        info = {}

        return observation, info

    def add_current_region_to_visited(self):
        """Add the current region to the set of visited regions."""
        region = self.current_position // self.region_size
        self.regions_visited.add(region)

    def _get_observation(self):
        """Get the current observation of the environment."""
        if not self.target_found:
            # During search phase, provide position relative to target
            rel_position = self.current_position - self.target
        else:
            # During rescue phase, provide position relative to base
            rel_position = self.current_position - self.base_point

        return {
            "relative_position": np.array([rel_position]),
            "direction": np.array([self.current_direction]),
            "farthest_right_rel": np.array(
                [self.farthest_right - self.current_position]
            ),
            "target_found": np.array([1 if self.target_found else 0]),
            "distance_to_base": np.array(
                [abs(self.current_position - self.base_point)]
            ),
            "search_phase": np.array([self.search_phase]),
        }

    def step(self, action):
        """
        Take a step in the environment based on the given action.

        Parameters:
            action (int): The action to take (0: left, 1: right)

        Returns:
            observation (dict): The new observation of the environment
            reward (float): The reward received for the action
            terminated (bool): Whether the episode is terminated
            truncated (bool): Whether the episode is truncated
            info (dict): Additional information about the step
        """
        self.steps_taken += 1
        self.previous_action = action

        # Track previous position for reward calculation
        previous_position = self.current_position

        # Update position based on action
        if action == 0:
            self.current_position -= 1
        else:
            self.current_position += 1

        # Keep agent within left boundary (can go infinitely to the right)
        self.current_position = max(0, self.current_position)

        # Update direction based on action
        self.current_direction = 0 if action == 0 else 1

        # Update tracking variables
        self.previous_positions.append(self.current_position)
        self.farthest_right = max(self.farthest_right, self.current_position)
        self.prev_target_position = self.target

        # Add current region to visited
        self.add_current_region_to_visited()

        # Initialize reward and info tracking
        reward = 0
        reward_components = {}
        terminated = False
        truncated = self.steps_taken >= self.max_steps

        # ===== REWARD SYSTEM =====

        # Base step penalty to encourage efficiency
        step_penalty = -0.1
        reward += step_penalty
        reward_components["step_penalty"] = step_penalty

        # Check mission phase and progress
        if not self.target_found:
            # === SEARCH PHASE ===

            # Update search phase if needed
            if (
                self.search_phase == self.SEARCH_PHASE_INITIAL
                and self.current_position >= self.initial_exploration_threshold
            ):
                self.search_phase = self.SEARCH_PHASE_RETURN
                print_info(
                    f"Initial exploration complete. Returning to base.",
                    category="SEARCH",
                    step=self.steps_taken,
                )
            elif (
                self.search_phase == self.SEARCH_PHASE_RETURN
                and self.current_position == 0
            ):
                self.search_phase = self.SEARCH_PHASE_EXPLORE
                print_info(
                    f"Returned to base. Starting full exploration to the right.",
                    category="SEARCH",
                    step=self.steps_taken,
                )

            # Check if target is found
            if abs(self.current_position - self.target) <= 1:
                # Major reward for finding the target
                target_found_reward = 50.0
                reward += target_found_reward
                reward_components["target_found"] = target_found_reward
                self.target_found = True
                print_info(
                    f"Target found at position {self.target}! Agent at position {self.current_position}",
                    category="SUCCESS",
                    step=self.steps_taken,
                    frequency=1,
                )
            else:
                # SEARCH PHASE REWARDS

                # Different rewards based on search phase
                if self.search_phase == self.SEARCH_PHASE_INITIAL:
                    # During initial exploration, reward going right
                    if action == 1:
                        right_reward = 0.3
                        reward += right_reward
                        reward_components["go_right"] = right_reward

                elif self.search_phase == self.SEARCH_PHASE_RETURN:
                    # During return phase, reward going left toward base
                    if action == 0:
                        return_reward = 0.4
                        reward += return_reward
                        reward_components["return_to_base"] = return_reward

                elif self.search_phase == self.SEARCH_PHASE_EXPLORE:
                    # During full exploration, reward going right and exploring new areas
                    if action == 1:
                        explore_reward = 0.3
                        reward += explore_reward
                        reward_components["explore_right"] = explore_reward

                    # Extra reward for reaching new maximum right position
                    if self.current_position > self.farthest_right - 1:
                        new_territory_reward = 0.4
                        reward += new_territory_reward
                        reward_components["new_territory"] = new_territory_reward

                # Reward for visiting new regions (all phases)
                if len(self.regions_visited) > self.steps_taken / 30:
                    region_reward = 0.2
                    reward += region_reward
                    reward_components["new_regions"] = region_reward
        else:
            # === RESCUE PHASE (after target found) ===

            # Check if rescue is complete
            if self.current_position == self.base_point:
                # Major reward for completing the rescue mission
                rescue_reward = 100.0
                reward += rescue_reward
                reward_components["rescue_complete"] = rescue_reward
                self.rescue_complete = True
                terminated = True
                print_info(
                    f"Rescue complete! Agent returned to base with target!",
                    category="SUCCESS",
                    step=self.steps_taken,
                    frequency=1,
                )
            else:
                # RESCUE PHASE REWARDS - Encourage efficient return to base

                # 1. Reward for moving toward base (position 0)
                if action == 0 and previous_position > self.current_position:
                    # Stronger reward the closer we get to base
                    base_progress = 0.5 * (1 / max(1, self.current_position / 10))
                    reward += base_progress
                    reward_components["base_progress"] = base_progress

                # 2. Penalty for moving away from base
                if action == 1:
                    away_penalty = -0.3
                    reward += away_penalty
                    reward_components["away_from_base"] = away_penalty

                # 3. Progress bonus based on proximity to base
                if previous_position > self.current_position:
                    proximity_bonus = (
                        0.2
                        * (previous_position - self.current_position)
                        / previous_position
                    )
                    reward += proximity_bonus
                    reward_components["proximity"] = proximity_bonus

        # Update target position if moving (only during search phase)
        if self.move_target and not self.target_found:
            if self.target_move_prob > 0:
                if self.np_random.random() < self.target_move_prob:
                    # Update zigzag pattern
                    self.zigzag_phase = (self.zigzag_phase + 1) % self.zigzag_period

                    # Calculate zigzag displacement using sine wave
                    # This creates a smooth zigzag pattern
                    zigzag_displacement = int(
                        self.zigzag_amplitude
                        * math.sin(2 * math.pi * self.zigzag_phase / self.zigzag_period)
                    )

                    # Calculate new target position based on zigzag pattern
                    new_target = self.zigzag_center + zigzag_displacement

                    # Move the center of the zigzag pattern slowly
                    if self.np_random.random() < 0.2:  # 20% chance to move the center
                        # Randomly choose direction for center movement
                        center_direction = self.np_random.choice([-1, 1])
                        self.zigzag_center += (
                            center_direction * self.np_random.integers(1, 3)
                        )

                    # Set the target position
                    self.target = new_target

                    # Ensure target stays within bounds (between 0 and target_range)
                    self.target = max(0, min(self.target, self.target_range))
                else:
                    # Target doesn't move this step
                    pass
            else:
                # No movement if probability is 0
                pass

        observation = self._get_observation()
        info = {
            "target_found": self.target_found,
            "rescue_complete": self.rescue_complete,
            "target_position": self.target,
            "regions_visited": len(self.regions_visited),
            "reward_components": reward_components,
            "search_phase": self.search_phase,
        }

        return observation, reward, terminated, truncated, info

    def render(self, mode="console"):
        """Render the environment."""
        if mode == "console":
            phase = "RESCUE" if self.target_found else "SEARCH"
            print(
                f"[{phase}] Position: {self.current_position}, Target: {self.target}, Base: {self.base_point}"
            )
        else:
            raise NotImplementedError("Render mode not supported")

    def close(self):
        """Close the environment."""
        pass
