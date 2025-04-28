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
PRINT_REWARDS = True  # Changed to True to show reward information
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
    - Moving targets with configurable movement pattern
    - Search strategy: go right for a bit, then to base, then explore right
    - Rescue: return to base with the targets
    - Sensing capability to detect nearby targets

    Parameters:
        max_steps (int): Maximum number of steps per episode
        target_range (int): Range for initial target placement
        region_size (int): Size of each region for tracking visitation
        move_target (bool): Whether the targets should move
        target_move_prob (float): Probability of targets moving on each step
        target_speed (int): Maximum distance targets can move in one step
        num_targets (int): Number of targets to place in the environment
        sensing_range (int): Range within which agent can sense target direction and distance
    """

    metadata = {"render.modes": ["console"]}

    def __init__(
        self,
        max_steps: int = 1000,
        target_range: int = 1000,  # Doubled from 500 to 1000
        region_size: int = 10,
        move_target: bool = False,
        target_move_prob: float = 0.1,
        target_speed: int = 1,
        num_targets: int = 2,  # Default to 2 targets now
        sensing_range: int = 50,  # New parameter for sensing nearby targets
        seed: Optional[int] = None,
    ):
        super().__init__()

        # Basic environment settings
        self.max_steps = max_steps
        self.target_range = target_range
        self.region_size = region_size
        self.num_targets = num_targets
        self.sensing_range = sensing_range

        # Target movement settings
        self.move_target = move_target
        self.target_move_prob = target_move_prob
        self.target_speed = target_speed

        # Zigzag pattern parameters
        self.zigzag_amplitude = 20  # Max distance from the center line
        self.zigzag_period = 30  # Steps to complete one full zigzag cycle
        self.zigzag_phases = [0] * num_targets  # Phase for each target
        self.zigzag_centers = [0] * num_targets  # Center for each target

        # Base point (rescue location) is at position 0
        self.base_point = 0

        # Search and rescue state
        self.targets_found = [False] * num_targets
        self.targets_rescued = [False] * num_targets
        self.target_found = False  # Any target found
        self.rescue_complete = False  # At least one target rescued
        self.all_targets_rescued = False  # All targets rescued

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
                "sensing_info": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
                ),  # [distance, direction] to nearest target within sensing range
                "targets_found_count": spaces.Discrete(
                    num_targets + 1
                ),  # Number of targets found
            }
        )

        # Initialize state
        self.current_position = 0
        self.targets = []  # List of (target_id, position) tuples
        self.previous_positions = []
        self.steps_taken = 0
        self.current_direction = 1  # Start facing right
        self.previous_action = None
        self.regions_visited = set()
        self.farthest_right = 0
        self.target_move_directions = [0] * num_targets

        # Initial exploration threshold
        self.initial_exploration_threshold = 150  # Go right about this far initially

        # Track repeated positions for stagnation detection
        self.position_history = []
        self.stagnation_threshold = (
            5  # Number of repeated positions to consider as stagnation
        )

        # Track movement direction history for momentum
        self.consecutive_right_moves = 0
        self.consecutive_left_moves = 0

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

        # Reset agent to starting position (at position 0)
        self.current_position = 0
        self.steps_taken = 0
        self.previous_action = None

        # Reset search and rescue state
        self.targets_found = [False] * self.num_targets
        self.targets_rescued = [False] * self.num_targets
        self.target_found = False
        self.rescue_complete = False
        self.all_targets_rescued = False
        self.search_phase = self.SEARCH_PHASE_INITIAL

        # Place targets (randomly between 0 and target_range)
        self.targets = []
        for i in range(self.num_targets):
            target_pos = self.np_random.integers(0, self.target_range)
            self.targets.append((i, target_pos))
            self.zigzag_phases[i] = 0
            self.zigzag_centers[i] = target_pos

        # Reset tracking variables
        self.regions_visited = set()
        self.farthest_right = self.current_position
        self.target_move_directions = [0] * self.num_targets

        # Reset positions history
        self.previous_positions = []

        # Reset position history for stagnation detection
        self.position_history = []

        # Reset momentum tracking
        self.consecutive_right_moves = 0
        self.consecutive_left_moves = 0

        # For exploration milestones
        self.next_milestone = 50  # First milestone at position 50

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
            # During search phase, provide position relative to nearest target
            nearest_target_pos = self._get_nearest_target_position()
            rel_position = self.current_position - nearest_target_pos
        else:
            # During rescue phase, provide position relative to base
            rel_position = self.current_position - self.base_point

        # Get sensing information - distance and direction to nearest target within range
        sensing_distance, sensing_direction = self._get_sensing_information()

        # Count how many targets have been found
        targets_found_count = sum(self.targets_found)

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
            "sensing_info": np.array([sensing_distance, sensing_direction]),
            "targets_found_count": np.array([targets_found_count]),
        }

    def _get_nearest_target_position(self):
        """Get the position of the nearest target."""
        if not self.targets:
            return 0

        # Find nearest target that hasn't been found yet
        nearest_dist = float("inf")
        nearest_pos = 0

        for i, (target_id, pos) in enumerate(self.targets):
            # Only consider targets that haven't been found
            if not self.targets_found[target_id]:
                dist = abs(self.current_position - pos)
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_pos = pos

        # If all targets are found, return agent position (to avoid influencing behavior)
        if nearest_dist == float("inf"):
            return self.current_position

        return nearest_pos

    def _get_sensing_information(self):
        """
        Get sensing information about nearby targets.

        Returns:
            Tuple of (distance, direction):
                - distance: distance to nearest target within sensing range, or -1 if none detected
                - direction: -1 for left, 1 for right, 0 if no target within range or targets in both directions
        """
        nearest_dist = float("inf")
        direction = 0

        for i, (target_id, pos) in enumerate(self.targets):
            # Only consider targets that haven't been found
            if not self.targets_found[target_id]:
                dist = abs(self.current_position - pos)

                # Check if target is within sensing range
                if dist <= self.sensing_range and dist < nearest_dist:
                    nearest_dist = dist
                    # Calculate direction: -1 for left, 1 for right
                    if pos < self.current_position:
                        direction = -1
                    else:
                        direction = 1

        # If no target within range, return -1 for distance
        if nearest_dist == float("inf"):
            return -1, 0

        return nearest_dist, direction

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
            # Update momentum tracking
            self.consecutive_left_moves += 1
            self.consecutive_right_moves = 0
        else:
            self.current_position += 1
            # Update momentum tracking
            self.consecutive_right_moves += 1
            self.consecutive_left_moves = 0

        # Keep agent within left boundary (can go infinitely to the right)
        self.current_position = max(0, self.current_position)

        # Update direction based on action
        self.current_direction = 0 if action == 0 else 1

        # Update tracking variables
        self.previous_positions.append(self.current_position)

        # Update farthest_right, but only when significantly farther
        if self.current_position > self.farthest_right + 3:
            self.farthest_right = self.current_position
            print_info(
                f"New farthest position: {self.farthest_right}",
                category="EXPLORATION",
                step=self.steps_taken,
            )

        # Add current region to visited
        self.add_current_region_to_visited()

        # Update position history for stagnation detection
        self.position_history.append(self.current_position)
        if len(self.position_history) > 10:
            self.position_history = self.position_history[-10:]

        # Initialize reward and info tracking
        reward = 0
        reward_components = {}
        terminated = False
        truncated = self.steps_taken >= self.max_steps

        # Calculate rewards using helper methods
        rescue_reward, rescue_components, rescue_terminated = (
            self._calculate_rescue_rewards()
        )
        target_finding_reward, target_finding_components = (
            self._calculate_target_finding_rewards()
        )
        directional_reward, directional_components = (
            self._calculate_directional_rewards(action)
        )

        # Apply all reward components
        reward += rescue_reward + target_finding_reward + directional_reward
        reward_components.update(rescue_components)
        reward_components.update(target_finding_components)
        reward_components.update(directional_components)

        # Set terminated flag if rescue terminated
        terminated = rescue_terminated

        # Update target positions if moving targets is enabled
        self._update_target_positions()

        # Print reward information to help debugging
        if PRINT_REWARDS:
            reward_info = ", ".join(
                [f"{k}: {v:.2f}" for k, v in reward_components.items()]
            )
            action_name = "LEFT" if action == 0 else "RIGHT"
            print_info(
                f"Action: {action} ({action_name}), Rewards: {reward_info} (total: {reward:.2f})",
                category="REWARD",
                step=self.steps_taken,
                frequency=PRINT_FREQUENCY,
            )

        observation = self._get_observation()
        info = {
            "target_found": self.target_found,
            "rescue_complete": self.rescue_complete,
            "all_targets_rescued": self.all_targets_rescued,
            "targets": self.targets,
            "targets_found": self.targets_found,
            "targets_rescued": self.targets_rescued,
            "regions_visited": len(self.regions_visited),
            "reward_components": reward_components,
            "search_phase": self.search_phase,
            "sensing_info": (
                observation["sensing_info"][0],
                observation["sensing_info"][1],
            ),
            "current_rewards": reward_components,  # Add rewards to info for UI display
            "total_reward": reward,  # Add total reward for UI display
        }

        return observation, reward, terminated, truncated, info

    def _calculate_rescue_rewards(self):
        """
        Calculate rewards for rescuing targets (bringing them to the base).

        Returns:
            tuple: (total_reward, reward_components, terminated)
        """
        reward = 0
        reward_components = {}
        terminated = False

        # Check for rescuing targets (bringing them to base)
        if any(self.targets_found) and self.current_position == self.base_point:
            # Calculate how many new targets are being rescued
            targets_being_rescued = 0
            for i, found in enumerate(self.targets_found):
                if found and not self.targets_rescued[i]:
                    targets_being_rescued += 1
                    self.targets_rescued[i] = True

            if targets_being_rescued > 0:
                # Total rescued so far
                total_rescued = sum(self.targets_rescued)

                # Big reward for rescuing targets (all at once)
                rescue_reward = 500.0 * targets_being_rescued
                reward += rescue_reward
                reward_components["rescue"] = rescue_reward

                # Bonus for completing the full rescue (all targets)
                if all(self.targets_rescued):
                    all_rescued_bonus = 1000.0
                    reward += all_rescued_bonus
                    reward_components["all_rescued_bonus"] = all_rescued_bonus
                    self.all_targets_rescued = True
                    self.rescue_complete = True
                    terminated = True
                    print_info(
                        f"Complete rescue! All {self.num_targets} targets rescued!",
                        category="SUCCESS",
                        step=self.steps_taken,
                        frequency=1,
                    )
                else:
                    # If not all rescued, encourage continuing exploration
                    print_info(
                        f"Partial rescue: {total_rescued}/{self.num_targets} targets rescued. Continue searching!",
                        category="PROGRESS",
                        step=self.steps_taken,
                        frequency=1,
                    )

        return reward, reward_components, terminated

    def _calculate_target_finding_rewards(self):
        """
        Calculate rewards for finding targets.

        Returns:
            tuple: (total_reward, reward_components)
        """
        reward = 0
        reward_components = {}

        for i, (target_id, target_pos) in enumerate(self.targets):
            if (
                abs(self.current_position - target_pos) <= 2
                and not self.targets_found[target_id]
            ):
                target_found_reward = 200.0
                reward += target_found_reward
                reward_components["target_found"] = target_found_reward
                self.targets_found[target_id] = True
                self.target_found = True

                # Count targets remaining to find
                targets_remaining = sum(1 for f in self.targets_found if not f)

                print_info(
                    f"Target {target_id} found at position {target_pos}! Targets remaining: {targets_remaining}",
                    category="SUCCESS",
                    step=self.steps_taken,
                    frequency=1,
                )
                break

        return reward, reward_components

    def _calculate_directional_rewards(self, action):
        """
        Calculate directional rewards based on the current situation.

        Args:
            action (int): The action taken (0: left, 1: right)

        Returns:
            tuple: (total_reward, reward_components)
        """
        reward = 0
        reward_components = {}

        # Only calculate directional rewards if no targets were found or rescued in this step
        if not self._just_found_target() and not self._just_rescued_target():
            # Count found but not rescued targets
            carrying_targets = self._count_carrying_targets()

            # Count unfound targets
            unfound_targets = self._count_unfound_targets()

            # Simple sensing of nearby targets
            sensing_distance, sensing_direction = self._get_sensing_information()
            sensed_target = sensing_distance > 0

            # Case 1: Carrying targets with none left to find - go left to base
            if carrying_targets > 0 and unfound_targets == 0:
                if action == 0:  # Moving left
                    return_reward = 1.0
                    reward += return_reward
                    reward_components["return_to_base"] = return_reward

            # Case 2: Following sensed targets
            elif sensed_target:
                # Moving in direction of sensed target
                if (sensing_direction == -1 and action == 0) or (
                    sensing_direction == 1 and action == 1
                ):
                    sensing_reward = 1.0
                    reward += sensing_reward
                    reward_components["follow_sensing"] = sensing_reward

            # Case 3: Default exploration behavior - go right
            else:
                # Reward exploring right strongly when targets still need to be found
                if unfound_targets > 0 and action == 1:
                    explore_reward = 1.0
                    reward += explore_reward
                    reward_components["explore_right"] = explore_reward

                    # Extra reward for sustained rightward movement
                    if self.consecutive_right_moves > 3:
                        momentum_reward = min(0.5, 0.1 * self.consecutive_right_moves)
                        reward += momentum_reward
                        reward_components["exploration_momentum"] = momentum_reward

            # Small living penalty to encourage efficiency (reduced)
            living_penalty = -0.5  # Reduced from -0.1 to -0.05
            reward += living_penalty
            reward_components["living_penalty"] = living_penalty

        return reward, reward_components

    def _just_found_target(self):
        """
        Check if a target was just found in this step.

        Returns:
            bool: True if a target was just found, False otherwise
        """
        # Can determine if a target was just found from reward components
        # This is a simplification - actual implementation depends on how you track this
        return (
            "target_found" in self._last_reward_components()
            if hasattr(self, "_last_reward_components")
            else False
        )

    def _just_rescued_target(self):
        """
        Check if a target was just rescued in this step.

        Returns:
            bool: True if a target was just rescued, False otherwise
        """
        # Can determine if a target was just rescued from reward components
        # This is a simplification - actual implementation depends on how you track this
        return (
            "rescue" in self._last_reward_components()
            if hasattr(self, "_last_reward_components")
            else False
        )

    def _count_carrying_targets(self):
        """
        Count the number of targets that have been found but not yet rescued.

        Returns:
            int: Number of targets being carried
        """
        return sum(
            1
            for i, found in enumerate(self.targets_found)
            if found and not self.targets_rescued[i]
        )

    def _count_unfound_targets(self):
        """
        Count the number of targets that have not been found yet.

        Returns:
            int: Number of unfound targets
        """
        return sum(1 for found in self.targets_found if not found)

    def _update_target_positions(self):
        """Update the positions of moving targets based on configured movement patterns."""
        if self.target_move_prob > 0:
            for i, (target_id, target_pos) in enumerate(self.targets):
                if self.targets_found[target_id]:
                    continue  # Skip found targets

                if self.np_random.random() < self.target_move_prob:
                    # Update zigzag pattern
                    self.zigzag_phases[i] = (
                        self.zigzag_phases[i] + 1
                    ) % self.zigzag_period

                    # Calculate zigzag displacement using sine wave
                    # This creates a smooth zigzag pattern
                    zigzag_displacement = int(
                        self.zigzag_amplitude
                        * math.sin(
                            2 * math.pi * self.zigzag_phases[i] / self.zigzag_period
                        )
                    )

                    # Calculate new target position based on zigzag pattern
                    new_target = self.zigzag_centers[i] + zigzag_displacement

                    # Move the center of the zigzag pattern slowly
                    if self.np_random.random() < 0.2:  # 20% chance to move the center
                        # Randomly choose direction for center movement
                        center_direction = self.np_random.choice([-1, 1])
                        self.zigzag_centers[
                            i
                        ] += center_direction * self.np_random.integers(1, 3)

                    # Set the target position
                    self.targets[i] = (target_id, new_target)

                    # Ensure target stays within bounds (between 0 and target_range)
                    target_pos = max(0, min(new_target, self.target_range))
                    self.targets[i] = (target_id, target_pos)

    def _last_reward_components(self):
        """
        Get the reward components from the last step.
        This is just a placeholder - you'll need to implement actual tracking.

        Returns:
            dict: Last step's reward components
        """
        # In a real implementation, you would track this between steps
        return {}

    def render(self, mode="console"):
        """Render the environment."""
        if mode == "console":
            phase = "RESCUE" if self.target_found else "SEARCH"
            phase_name = ["INITIAL", "RETURN", "EXPLORE"][self.search_phase]
            targets_str = ", ".join(
                [
                    f"{id}:{pos}{'(F)' if self.targets_found[id] else ''}"
                    for id, pos in self.targets
                ]
            )
            sensing_dist, sensing_dir = self._get_sensing_information()
            sensing_str = f"Sensing: {'None' if sensing_dist < 0 else f'Dist={sensing_dist}, Dir={sensing_dir}'}"

            # Add reward and action information to the display
            last_step_rewards = "No rewards yet"
            last_action = "None"
            if hasattr(self, "previous_rewards") and hasattr(self, "previous_action"):
                last_step_rewards = ", ".join(
                    [f"{k}: {v:.2f}" for k, v in self.previous_rewards.items()]
                )
                last_action = "LEFT" if self.previous_action == 0 else "RIGHT"

            print(
                f"[{phase}:{phase_name}] Position: {self.current_position}, Targets: {targets_str}, "
                f"Base: {self.base_point}, {sensing_str}\n"
                f"Last Action: {last_action}, Rewards: {last_step_rewards}"
            )
        else:
            raise NotImplementedError("Render mode not supported")

    def close(self):
        """Close the environment."""
        pass
