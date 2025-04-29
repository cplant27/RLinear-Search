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


def print_reward_summary(
    step: int, action: int, reward_components: Dict[str, float], total_reward: float
):
    """Print a standardized reward summary format."""
    # Convert action number to direction string
    if action == 0:
        action_name = "LEFT"
    elif action == 1:
        action_name = "RIGHT"
    elif action == 2:
        action_name = "SIGNAL"
    else:
        action_name = str(action)

    # Format the reward components as a string
    reward_breakdown = " | ".join(
        f"{k}: {v:.2f}" for k, v in reward_components.items() if v != 0
    )

    # Print in the format: step, action, total reward, reward breakdown
    print(
        f"Step {step:4d} | Action: {action_name} | Total: {total_reward:.2f} | {reward_breakdown}"
    )


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

        # Action space: 0 = left, 1 = right, 2 = signal return to base
        self.action_space = spaces.Discrete(3)  # Expanded from 2 to 3 actions

        # Observation space
        self.observation_space = spaces.Dict(
            {
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

        # Optimal algorithm tracking
        self.optimal_steps = 0
        self.optimal_distance = 0
        self.optimal_path = []

        # Reward tracking
        self.episode_rewards = {
            "step_penalty": 0,
            "target_found": 0,
            "rescue_complete": 0,
            "go_right": 0,
            "return_to_base": 0,
            "start_explore": 0,
            "explore_right": 0,
            "new_territory": 0,
            "new_regions": 0,
            "base_progress": 0,
            "away_from_base": 0,
            "wrong_direction": 0,
            "oscillation": 0,
            "stagnation": 0,
            "explore_progress": 0,
            "back_penalty": 0,
            "distance_bonus": 0,
            "phase_transition": 0,
            "redundant_signal": 0,
            "explore_signal_penalty": 0,
        }

        # Performance metrics
        self.metrics = {
            "steps_to_find_target": 0,
            "steps_to_rescue": 0,
            "total_distance_traveled": 0,
            "competitive_ratio": 0.0,
            "optimal_steps": 0,
            "optimal_distance": 0,
        }

        # Initial exploration threshold
        self.initial_exploration_threshold = 150  # Go right about this far initially

        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        """Set the random seed for the environment."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def calculate_optimal_path(self):
        """Calculate the optimal path to find and rescue the target."""
        # For a stationary target, optimal path is:
        # 1. Go directly to target position
        # 2. Return to base
        optimal_steps = abs(self.target) * 2  # Steps to target and back
        optimal_distance = abs(self.target) * 2  # Distance to target and back

        # For moving target, we need to consider the zigzag pattern
        if self.move_target:
            # Calculate the maximum distance the target can move
            max_target_movement = self.zigzag_amplitude * 2  # Full zigzag cycle
            # Add some buffer for target movement
            optimal_steps += max_target_movement
            optimal_distance += max_target_movement

        return optimal_steps, optimal_distance

    def update_metrics(self):
        """Update performance metrics."""
        # Calculate total distance traveled
        total_distance = sum(
            abs(self.previous_positions[i] - self.previous_positions[i - 1])
            for i in range(1, len(self.previous_positions))
        )

        # Update metrics
        self.metrics.update(
            {
                "steps_to_find_target": self.steps_taken if self.target_found else 0,
                "steps_to_rescue": self.steps_taken if self.rescue_complete else 0,
                "total_distance_traveled": total_distance,
                "optimal_steps": self.optimal_steps,
                "optimal_distance": self.optimal_distance,
            }
        )

        # Calculate competitive ratio
        if self.optimal_steps > 0:
            self.metrics["competitive_ratio"] = self.steps_taken / self.optimal_steps
        else:
            self.metrics["competitive_ratio"] = float("inf")

    def reset(self, seed=None, options=None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state."""
        if seed is not None:
            super().reset(seed=seed)
            np.random.seed(seed)
            random.seed(seed)

        # Reset agent to starting position (around 100)
        self.current_position = self.np_random.integers(80, 120)
        self.initial_agent_position = self.current_position  # Store initial position
        self.steps_taken = 0
        self.previous_action = None

        # Reset search and rescue state
        self.target_found = False
        self.rescue_complete = False
        self.search_phase = self.SEARCH_PHASE_INITIAL

        # Place target (randomly between 0 and target_range)
        self.target = self.np_random.integers(0, self.target_range)
        self.initial_target_position = self.target  # Store initial target position

        # Calculate optimal path
        self.optimal_steps, self.optimal_distance = self.calculate_optimal_path()

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

        # Reset reward tracking
        for key in self.episode_rewards:
            self.episode_rewards[key] = 0

        # Reset metrics
        for key in self.metrics:
            self.metrics[key] = 0

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
        # Calculate observation components with correct types
        distance_to_base = np.array(
            [float(self.current_position)], dtype=np.float32
        )  # Base is at 0
        farthest_right_rel = np.array(
            [float(self.current_position - self.farthest_right)],
            dtype=np.float32,  # Correct calculation
        )

        obs = {
            "direction": int(self.current_direction),  # int for Discrete(2)
            "farthest_right_rel": farthest_right_rel,  # float32 array shape (1,)
            "target_found": int(self.target_found),  # int for Discrete(2)
            "distance_to_base": distance_to_base,  # float32 array shape (1,)
            "search_phase": int(self.search_phase),  # int for Discrete(3)
        }
        return obs

    def get_reward_summary(self) -> Dict[str, float]:
        """Get a summary of rewards earned during the episode."""
        return {
            "total_reward": sum(self.episode_rewards.values()),
            **self.episode_rewards,
        }

    def step(self, action):
        """
        Take a step in the environment based on the given action.

        Parameters:
            action (int): The action to take (0: left, 1: right, 2: signal return to base)

        Returns:
            observation (dict): The new observation of the environment
            reward (float): The reward received for the action
            terminated (bool): Whether the episode is terminated
            truncated (bool): Whether the episode is truncated
            info (dict): Additional information about the step
        """
        # Reward constants
        # Major Milestone Rewards
        TARGET_FOUND_REWARD = 500.0
        RESCUE_COMPLETE_REWARD = 1000.0
        # Phase Transition Rewards
        PHASE_TRANSITION_REWARD_OPTIMAL = 1.0
        PHASE_TRANSITION_REWARD_SUBOPTIMAL = 0.5
        # Exploration Rewards
        START_EXPLORE_REWARD = 0.5
        EXPLORE_PROGRESS_REWARD = 0.5
        NEW_REGIONS_REWARD = 0.2
        # Navigation Rewards
        GO_RIGHT_REWARD = 0.3
        RETURN_TO_BASE_REWARD = 0.4
        BASE_PROGRESS_REWARD = 0.5
        # Movement Penalties
        STEP_PENALTY = -0.1
        WRONG_DIRECTION_PENALTY_INITIAL = -0.3
        WRONG_DIRECTION_PENALTY_RETURN = -0.4
        BACK_PENALTY = -0.3
        OSCILLATION_PENALTY = -5.0
        # Signal Penalties
        REDUNDANT_SIGNAL_PENALTY = -0.2
        EXPLORE_SIGNAL_PENALTY = -0.3
        # Base-related Penalties
        AWAY_FROM_BASE_PENALTY_BASE = -1.0
        AWAY_FROM_BASE_PENALTY_MULTIPLIER = -0.05

        self.steps_taken += 1
        self.previous_action = action

        # Track previous position for reward calculation
        previous_position = self.current_position

        # Initialize reward and info tracking
        reward = 0
        reward_components = {}
        terminated = False
        truncated = self.steps_taken >= self.max_steps

        # Process the "signal return to base" action
        if action == 2:
            # Only transition if we're in the initial exploration phase
            if self.search_phase == self.SEARCH_PHASE_INITIAL:
                # Increase the reward based on distance explored - the further the agent has gone,
                # the better the timing of the return signal
                exploration_distance = max(
                    0, self.current_position - 100
                )  # Starting around 100

                # Higher reward for optimal distance range (around 150-250)
                if 150 <= exploration_distance <= 250:
                    phase_transition_reward = PHASE_TRANSITION_REWARD_OPTIMAL  # Optimal range gets higher reward
                else:
                    # Gradually less reward for going too little or too far
                    phase_transition_reward = PHASE_TRANSITION_REWARD_SUBOPTIMAL

                # Change phase to return to base
                self.search_phase = self.SEARCH_PHASE_RETURN
                print_info(
                    f"Agent decided to return to base from position {self.current_position}.",
                    category="SEARCH",
                    step=self.steps_taken,
                )

                # Add the phase transition reward
                reward += phase_transition_reward
                reward_components["phase_transition"] = phase_transition_reward
                self.episode_rewards["phase_transition"] = (
                    self.episode_rewards.get("phase_transition", 0)
                    + phase_transition_reward
                )
            elif self.search_phase == self.SEARCH_PHASE_RETURN:
                # If already in return phase, small penalty for redundant signaling
                reward += REDUNDANT_SIGNAL_PENALTY
                reward_components["redundant_signal"] = REDUNDANT_SIGNAL_PENALTY
                self.episode_rewards["redundant_signal"] = (
                    self.episode_rewards.get("redundant_signal", 0)
                    + REDUNDANT_SIGNAL_PENALTY
                )
            elif self.search_phase == self.SEARCH_PHASE_EXPLORE:
                # If in explore phase, penalty for signaling (should be exploring at this point)
                reward += EXPLORE_SIGNAL_PENALTY
                reward_components["explore_signal_penalty"] = EXPLORE_SIGNAL_PENALTY
                self.episode_rewards["explore_signal_penalty"] = (
                    self.episode_rewards.get("explore_signal_penalty", 0)
                    + EXPLORE_SIGNAL_PENALTY
                )

            # Instead of not moving, allow a slight left movement to encourage return to base
            # This avoids getting stuck in a loop of signaling while maintaining the intention
            self.current_position -= 1
            self.current_direction = 0  # Left direction
            action = 0  # Treat as left action for reward calculations

        # Update position and direction based on action (only 0 and 1 affect movement)
        if action == 0:
            self.current_position -= 1
            self.current_direction = 0
        elif action == 1:
            self.current_position += 1
            self.current_direction = 1
        # Action 2 (signal) already handled above

        # Keep agent within left boundary (can go infinitely to the right)
        self.current_position = max(0, self.current_position)

        # Update tracking variables
        self.previous_positions.append(self.current_position)
        self.farthest_right = max(self.farthest_right, self.current_position)
        self.prev_target_position = self.target

        # Add current region to visited
        self.add_current_region_to_visited()

        # ===== REWARD SYSTEM =====

        # Base step penalty to encourage efficiency
        reward += STEP_PENALTY
        reward_components["step_penalty"] = STEP_PENALTY
        self.episode_rewards["step_penalty"] += STEP_PENALTY

        # Check mission phase and progress
        if not self.target_found:
            # === SEARCH PHASE ===

            # REMOVE this automatic transition - let the agent decide
            # if (
            #     self.search_phase == self.SEARCH_PHASE_INITIAL
            #     and self.current_position >= self.initial_exploration_threshold
            # ):
            #     self.search_phase = self.SEARCH_PHASE_RETURN
            #     print_info(
            #         f"Initial exploration complete. Returning to base.",
            #         category="SEARCH",
            #         step=self.steps_taken,
            #     )

            # Transition
            if (
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
                reward += TARGET_FOUND_REWARD
                reward_components["target_found"] = TARGET_FOUND_REWARD
                self.episode_rewards["target_found"] += TARGET_FOUND_REWARD
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
                        reward += GO_RIGHT_REWARD
                        reward_components["go_right"] = GO_RIGHT_REWARD
                        self.episode_rewards["go_right"] += GO_RIGHT_REWARD
                    # Penalize moving left during initial exploration phase
                    elif action == 0 and self.current_position > 0:
                        reward += WRONG_DIRECTION_PENALTY_INITIAL
                        reward_components["wrong_direction"] = (
                            WRONG_DIRECTION_PENALTY_INITIAL
                        )
                        if "wrong_direction" not in self.episode_rewards:
                            self.episode_rewards["wrong_direction"] = 0
                        self.episode_rewards[
                            "wrong_direction"
                        ] += WRONG_DIRECTION_PENALTY_INITIAL

                elif self.search_phase == self.SEARCH_PHASE_RETURN:
                    # During return phase, reward going left toward base
                    if action == 0:
                        reward += RETURN_TO_BASE_REWARD
                        reward_components["return_to_base"] = RETURN_TO_BASE_REWARD
                        self.episode_rewards["return_to_base"] += RETURN_TO_BASE_REWARD
                    # Special case: when at base (position 0), encourage moving right to start full exploration
                    elif action == 1 and self.current_position == 0:
                        reward += START_EXPLORE_REWARD
                        reward_components["start_explore"] = START_EXPLORE_REWARD
                        self.episode_rewards["start_explore"] += START_EXPLORE_REWARD
                    # Penalize moving right when supposed to be returning to base
                    elif action == 1 and self.current_position > 0:
                        reward += WRONG_DIRECTION_PENALTY_RETURN
                        reward_components["wrong_direction"] = (
                            WRONG_DIRECTION_PENALTY_RETURN
                        )
                        if "wrong_direction" not in self.episode_rewards:
                            self.episode_rewards["wrong_direction"] = 0
                        self.episode_rewards[
                            "wrong_direction"
                        ] += WRONG_DIRECTION_PENALTY_RETURN

                elif self.search_phase == self.SEARCH_PHASE_EXPLORE:
                    # EXPLORATION PHASE REWARDS - Simple reward for moving right

                    # Reward for moving right, regardless of whether it's new territory
                    if action == 1 and self.current_position > previous_position:
                        # Simple fixed reward for exploring right
                        reward += EXPLORE_PROGRESS_REWARD
                        reward_components["explore_progress"] = EXPLORE_PROGRESS_REWARD
                        self.episode_rewards["explore_progress"] = (
                            self.episode_rewards.get("explore_progress", 0)
                            + EXPLORE_PROGRESS_REWARD
                        )

                    # Penalty for moving toward base (left)
                    if action == 0 and self.current_position < previous_position:
                        reward += BACK_PENALTY
                        reward_components["back_penalty"] = BACK_PENALTY
                        self.episode_rewards["back_penalty"] = (
                            self.episode_rewards.get("back_penalty", 0) + BACK_PENALTY
                        )

                # Add oscillation penalty to prevent getting stuck
                if len(self.previous_positions) >= 5:
                    # Check for oscillation pattern in last 5 positions
                    last_positions = self.previous_positions[-5:]

                    # Look for alternating pattern like A-B-A-B-A
                    oscillation_detected = (
                        last_positions[0] == last_positions[2] == last_positions[4]
                        and last_positions[1] == last_positions[3]
                        and last_positions[0] != last_positions[1]
                    )

                    # Or pattern like A-B-A-B where last 4 positions show alternating
                    if not oscillation_detected and len(self.previous_positions) >= 4:
                        last_four = self.previous_positions[-4:]
                        oscillation_detected = (
                            last_four[0] == last_four[2]
                            and last_four[1] == last_four[3]
                            and last_four[0] != last_four[1]
                        )

                    if oscillation_detected:
                        # Apply a strong oscillation penalty to break the loop
                        reward += OSCILLATION_PENALTY
                        reward_components["oscillation"] = OSCILLATION_PENALTY
                        if "oscillation" not in self.episode_rewards:
                            self.episode_rewards["oscillation"] = 0
                        self.episode_rewards["oscillation"] += OSCILLATION_PENALTY

                # Reward for visiting new regions (all phases)
                if len(self.regions_visited) > self.steps_taken / 30:
                    reward += NEW_REGIONS_REWARD
                    reward_components["new_regions"] = NEW_REGIONS_REWARD
                    self.episode_rewards["new_regions"] += NEW_REGIONS_REWARD

        else:
            # === RESCUE PHASE (after target found) ===

            # Check if rescue is complete
            if self.current_position == self.base_point:
                # Major reward for completing the rescue mission
                reward += RESCUE_COMPLETE_REWARD
                reward_components["rescue_complete"] = RESCUE_COMPLETE_REWARD
                self.episode_rewards["rescue_complete"] += RESCUE_COMPLETE_REWARD
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
                    # Simple fixed reward for moving toward base
                    reward += BASE_PROGRESS_REWARD
                    reward_components["base_progress"] = BASE_PROGRESS_REWARD
                    self.episode_rewards["base_progress"] += BASE_PROGRESS_REWARD

                # 2. Significantly increase penalty for moving away from base during rescue
                if action == 1:
                    # Stronger penalty the further we move from base
                    # Also increases based on steps taken to add urgency
                    away_penalty = AWAY_FROM_BASE_PENALTY_BASE + (
                        AWAY_FROM_BASE_PENALTY_MULTIPLIER * self.steps_taken / 100
                    )
                    reward += away_penalty
                    reward_components["away_from_base"] = away_penalty
                    self.episode_rewards["away_from_base"] += away_penalty

                # 4. Oscillation penalty during rescue phase
                if len(self.previous_positions) >= 5:
                    # Check for oscillation pattern in last 5 positions
                    last_positions = self.previous_positions[-5:]

                    # Look for alternating pattern like A-B-A-B-A
                    oscillation_detected = (
                        last_positions[0] == last_positions[2] == last_positions[4]
                        and last_positions[1] == last_positions[3]
                        and last_positions[0] != last_positions[1]
                    )

                    # Or pattern like A-B-A-B where last 4 positions show alternating
                    if not oscillation_detected and len(self.previous_positions) >= 4:
                        last_four = self.previous_positions[-4:]
                        oscillation_detected = (
                            last_four[0] == last_four[2]
                            and last_four[1] == last_four[3]
                            and last_four[0] != last_four[1]
                        )

                    if oscillation_detected:
                        # Apply a strong oscillation penalty to break the loop
                        reward += OSCILLATION_PENALTY
                        reward_components["oscillation"] = OSCILLATION_PENALTY
                        if "oscillation" not in self.episode_rewards:
                            self.episode_rewards["oscillation"] = 0
                        self.episode_rewards["oscillation"] += OSCILLATION_PENALTY

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

        # Print reward summary using the utility function
        print_reward_summary(self.steps_taken, action, reward_components, reward)

        # Update metrics at the end of the step
        self.update_metrics()

        observation = self._get_observation()

        # Create a comprehensive training summary
        training_summary = None
        if terminated or truncated:
            training_summary = {
                "rewards": self.get_reward_summary(),
                "performance": {
                    "steps": {
                        "total": self.steps_taken,
                        "to_find_target": self.metrics["steps_to_find_target"],
                        "to_rescue": self.metrics["steps_to_rescue"],
                        "optimal": self.metrics["optimal_steps"],
                    },
                    "distance": {
                        "total": self.metrics["total_distance_traveled"],
                        "optimal": self.metrics["optimal_distance"],
                    },
                    "competitive_ratio": self.metrics["competitive_ratio"],
                    "success": {
                        "target_found": self.target_found,
                        "rescue_complete": self.rescue_complete,
                    },
                },
            }

        info = {
            "target_found": self.target_found,
            "rescue_complete": self.rescue_complete,
            "target_position": self.target,
            "regions_visited": len(self.regions_visited),
            "reward_components": reward_components,
            "search_phase": self.search_phase,
            "training_summary": training_summary,
        }

        return observation, reward, terminated, truncated, info

    def render(self, mode="console"):
        """Render the environment."""
        if mode == "console":
            phase = "RESCUE" if self.target_found else "SEARCH"
            search_phase_name = (
                "INITIAL"
                if self.search_phase == self.SEARCH_PHASE_INITIAL
                else (
                    "RETURN"
                    if self.search_phase == self.SEARCH_PHASE_RETURN
                    else "EXPLORE"
                )
            )
            phase_info = (
                f"{phase} ({search_phase_name})" if not self.target_found else phase
            )
            print(
                f"[{phase_info}] Position: {self.current_position}, Target: {self.target}, Base: {self.base_point}"
            )
        else:
            raise NotImplementedError("Render mode not supported")

    def close(self):
        """Close the environment."""
        pass

    def calculate_custom_competitive_ratio(self):
        """Calculate competitive ratio using different formulas based on target position:
        - If target is to the right of agent: steps_taken/(2*target_spawn_loc - agent_spawn_loc)
        - If target is to the left of agent: steps_taken/agent_spawn_loc
        """
        # Case 1: Target is to the right of agent
        if self.initial_target_position > self.initial_agent_position:
            denominator = 2 * self.initial_target_position - self.initial_agent_position
            if denominator > 0:
                return self.steps_taken / denominator
            else:
                return float("inf")  # Handle the case when denominator is non-positive
        # Case 2: Target is to the left of agent
        else:
            agent_spawn_dist = abs(self.initial_agent_position)
            if agent_spawn_dist > 0:
                return self.steps_taken / agent_spawn_dist
            else:
                return float("inf")  # Handle the case when agent is at position 0

    def print_competitive_ratio(self):
        """Print a summary of the competitive ratio at the end of the episode."""
        standard_ratio = self.metrics["competitive_ratio"]
        custom_ratio = self.calculate_custom_competitive_ratio()

        print("\n" + "=" * 80)
        print(f"EPISODE SUMMARY:")
        print(f"Initial agent position: {self.initial_agent_position}")
        print(f"Initial target position: {self.initial_target_position}")
        print(f"Steps taken: {self.steps_taken}")
        print(f"Optimal path (calculated): {self.optimal_steps}")

        # Display formula based on target position
        if self.initial_target_position > self.initial_agent_position:
            print(f"Custom formula: steps_taken/(2*target_spawn_loc - agent_spawn_loc)")
            print(
                f"Custom formula denominator: {2 * self.initial_target_position - self.initial_agent_position}"
            )
        else:
            print(f"Custom formula: steps_taken/agent_spawn_dist")
            print(f"Custom formula denominator: {abs(self.initial_agent_position)}")

        print(f"Custom competitive ratio: {custom_ratio:.2f}")
        print(f"Standard competitive ratio: {standard_ratio:.2f}")
        print("=" * 80 + "\n")
