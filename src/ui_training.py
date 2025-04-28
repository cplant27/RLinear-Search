import random
import time
import tkinter as tk
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from src.environment import InfiniteLinearSearchEnv
from src.qlearning import observation_to_state
from src.ui_utils import save_q_table
from src.ui_visualization import EnvironmentVisualizer


def train_environment(
    env: InfiniteLinearSearchEnv,
    Q: Dict,
    visualizer: EnvironmentVisualizer,
    variables: Dict[str, tk.StringVar],
    root: tk.Tk,
    actions: List[int],
    num_episodes: int = 1000,
    alpha: float = 0.6,
    gamma: float = 0.99,
    epsilon: float = 0.8,
):
    """Run training with visualization for the specified number of episodes."""
    efficiency_ratios = []  # Track efficiency ratios
    exploration_coverages = [0.05]  # Start with 5% coverage assumption

    # Set learning parameters
    current_alpha = alpha
    current_epsilon = epsilon

    # Only clear if not already populated
    if not Q:
        # Default Q-value factory
        def default_q_values():
            return {action: 0.0 for action in actions}

        # Update as defaultdict
        Q.update(defaultdict(default_q_values))

    # Track training metrics
    best_performance = {
        "steps": float("inf"),
        "found": False,
        "search_score": 0,
    }

    # Training-specific variables
    reward_categories = {
        "total": 0,
        "new_position": 0,
        "new_region": 0,
        "target_found": 0,
    }

    for episode in range(num_episodes):
        obs, _ = env.reset()
        state = observation_to_state(obs)
        steps = 0
        done = False
        truncated = False
        episode_reward = 0
        visualizer.last_search_phase = -1  # Reset phase tracking for new episode

        # Reset markers for this episode
        visualizer.clear_markers()

        # Reset path tracking
        visualizer.reset_path()

        # Reset visible range for new episode
        visualizer.current_min_visible = -200
        visualizer.current_max_visible = 200
        visualizer.update_grid()

        # Create new target markers
        visualizer.create_target_markers(env.targets, visualizer.canvas)

        # Reset agent position in UI
        agent_x = visualizer.x_to_canvas(env.current_position)

        # Update position labels
        variables["current_pos"].set(f"Agent Position: {env.current_position}")
        targets_info = ", ".join([f"Target {id}: {pos}" for id, pos in env.targets])
        variables["target_pos"].set(f"Targets: {targets_info}")

        # Update parameter display
        variables["episode"].set(f"Episode: {episode}/{num_episodes}")
        variables["epsilon"].set(f"ε: {current_epsilon:.3f}")
        variables["regions"].set(f"Regions: {len(env.regions_visited)}")

        # Update info panel with initial status
        visualizer.update_info_panel(
            variables,
            steps=steps,
            reward=0.0,
            search_phase=env.search_phase,
            target_found=env.target_found,
        )

        variables["range"].set(
            f"Visible Range: [{visualizer.current_min_visible}, {visualizer.current_max_visible}]"
        )

        # Force UI update
        root.update_idletasks()

        while not done and not truncated:
            # Get current observation and convert to state
            current_obs = env._get_observation()
            state = observation_to_state(current_obs)

            # Choose action based on epsilon-greedy
            if random.random() < current_epsilon:
                # Simple exploration strategy
                action = random.choice(actions)
            else:
                # Exploit learned policy
                action = max(Q[state], key=Q[state].get)

            # Take action in environment
            next_obs, reward, done, truncated, info = env.step(action)
            next_state = observation_to_state(next_obs)
            episode_reward += reward

            # Categorize rewards
            reward_categories["total"] += reward
            if "reward_components" in info:
                for category, value in info["reward_components"].items():
                    if category in reward_categories:
                        reward_categories[category] += value

            # Check if visible range needs updating
            range_changed = visualizer.update_visible_range(
                env.current_position, env.targets
            )
            if range_changed:
                # Update grid
                visualizer.update_grid()

            # Standard Q-learning update
            best_next = max(Q[next_state].values())
            Q[state][action] += current_alpha * (
                reward + gamma * best_next - Q[state][action]
            )

            # Update the agent and targets markers
            agent_x = visualizer.update_agent_position(env.current_position)
            visualizer.update_targets_positions(env.targets, env.targets_found)

            # Update phases based on search phase
            if info.get("search_phase", 0) != visualizer.last_search_phase:
                visualizer.last_search_phase = info.get("search_phase", 0)
                visualizer.add_phase_marker(agent_x, visualizer.last_search_phase)

            # Move to next state
            state = next_state
            steps += 1

            # Update info panel
            visualizer.update_info_panel(
                variables,
                steps=steps,
                reward=episode_reward,
                search_phase=info.get("search_phase", 0),
                target_found=env.target_found,
            )

            # Update info labels
            variables["current_pos"].set(f"Agent Position: {env.current_position}")
            targets_info = ", ".join([f"Target {id}: {pos}" for id, pos in env.targets])
            variables["target_pos"].set(f"Targets: {targets_info}")
            variables["range"].set(
                f"Visible Range: [{visualizer.current_min_visible}, {visualizer.current_max_visible}]"
            )

            # Update region count
            regions_visited_count = len(env.regions_visited)
            variables["visit_count"].set(f"Regions visited: {regions_visited_count}")
            variables["regions"].set(f"Regions: {regions_visited_count}")

            # Only update UI every few steps for better performance
            if steps % 5 == 0:
                root.update()

            # Detect terminal state
            if steps >= env.max_steps:
                truncated = True

            if done or truncated:
                # Highlight success when target is found
                if env.target_found and env.rescue_complete:
                    visualizer.highlight_success(True)
                    root.update()  # Force update

                    # Update best performance
                    if steps < best_performance["steps"]:
                        best_performance["steps"] = steps
                        best_performance["found"] = True

                    # Calculate competitive ratio (efficiency)
                    # Get distance to the farthest target for ratio calculation
                    farthest_target_dist = 0
                    for _, pos in env.targets:
                        farthest_target_dist = max(farthest_target_dist, abs(pos - 0))

                    competitive_ratio = steps / max(1, farthest_target_dist)
                    efficiency_ratios.append(competitive_ratio)
                    if len(efficiency_ratios) > 20:
                        efficiency_ratios.pop(0)

                    # Show success
                    found_targets = sum(env.targets_found)
                    variables["status"].set(
                        f"Mission complete in {steps} steps! Found {found_targets} targets. Reward: {episode_reward:.1f}"
                    )
                elif env.target_found:
                    found_targets = sum(env.targets_found)
                    variables["status"].set(
                        f"Target(s) found ({found_targets}) but return failed. Steps: {steps}, Reward: {episode_reward:.1f}"
                    )
                else:
                    variables["status"].set(
                        f"Episode {episode}/{num_episodes}: Steps: {steps}, Reward: {episode_reward:.1f}"
                    )

                # Calculate search score based on regions visited
                search_score = len(env.regions_visited) * 2
                if search_score > best_performance["search_score"]:
                    best_performance["search_score"] = search_score

                # End of episode diagnostic output
                if episode % 10 == 0:
                    print(
                        f"Episode {episode}: Steps: {steps}, Reward: {episode_reward:.1f}"
                    )
                    print(f"Regions visited: {len(env.regions_visited)}")
                    if env.rescue_complete:
                        found_targets = sum(env.targets_found)
                        print(
                            f"MISSION COMPLETE! Return to base successful with {found_targets} targets."
                        )
                    elif env.target_found:
                        found_targets = [
                            i for i, found in enumerate(env.targets_found) if found
                        ]
                        print(f"TARGET(S) FOUND {found_targets} but return failed")
                    else:
                        print(f"Maximum steps reached")

                # Force UI update
                root.update()
                # Short pause between episodes
                time.sleep(0.5)
                break

        # Adapt learning parameters based on performance
        # Reduce exploration as training progresses
        current_epsilon = max(0.1, current_epsilon * 0.99)
        variables["epsilon"].set(f"ε: {current_epsilon:.3f}")

        # Decay learning rate
        current_alpha = max(0.1, current_alpha * 0.995)

        # Every 50 episodes, save Q-table
        if episode % 50 == 0 or episode == num_episodes - 1:
            save_q_table(Q, episode)
            variables["status"].set(f"Q-table saved at episode {episode}")
            root.update()

    # After training, show success
    if best_performance["found"]:
        variables["status"].set(
            f"Training complete! Best performance: {best_performance['steps']} steps, "
            f"Best search score: {best_performance['search_score']}"
        )
    else:
        variables["status"].set(
            f"Training complete. Mission not completed in any episode. "
            f"Best search score: {best_performance['search_score']}"
        )

    # Update UI
    root.update()
