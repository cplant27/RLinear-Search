import random
import tkinter as tk
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.environment import InfiniteLinearSearchEnv
from src.qlearning import observation_to_state
from src.ui_components import (
    create_agent_marker,
    create_canvas,
    create_info_frame,
    create_legend,
    create_params_frame,
    create_path_line,
    create_target_marker,
)
from src.ui_training import train_environment
from src.ui_utils import load_q_table
from src.ui_visualization import EnvironmentVisualizer, update_step


def test_policy_ui(
    env: InfiniteLinearSearchEnv,
    Q: Dict[Tuple, Dict[int, float]],
    actions: List[int],
    max_steps: int = 5000,
    delay: int = 0,
    mode: str = "test",
    num_episodes: int = 1,
    alpha: float = 0.6,
    gamma: float = 0.99,
    epsilon: float = 0.8,
    num_test_rounds: int = 1,  # Number of test rounds to run
) -> None:
    """
    Test or train the learned policy for search and rescue using a Tkinter UI.

    The canvas displays:
      - A red circle representing the agent.
      - A green circle representing the target (which can move).
      - Visual indicators of the agent's path.
      - Markers for phase transitions.

    The agent's movement is updated every 'delay' milliseconds.

    Args:
        env: The environment
        Q: Q-table for action selection
        actions: Available actions
        max_steps: Maximum steps per episode
        delay: Delay between steps in milliseconds
        mode: "test" for testing, "train" for training
        num_episodes: Number of episodes for training
        alpha: Learning rate for training
        gamma: Discount factor for training
        epsilon: Exploration rate for training
        num_test_rounds: Number of test rounds to run (default: 1)
    """
    # Set up canvas dimensions
    canvas_width = 1200
    canvas_height = 300

    # Create the main window
    root = tk.Tk()
    root.title("Search and Rescue - Q-learning Visualization")
    root.geometry(f"{canvas_width}x{canvas_height + 200}")

    # Track competitive ratios for multiple test rounds
    competitive_ratios = []
    current_test_round = [0]  # Use list to make it mutable in nested functions

    # Create StringVars for UI labels
    variables = {
        "status": tk.StringVar(value="Initializing..."),
        "current_pos": tk.StringVar(value="Agent Position: 0"),
        "target_pos": tk.StringVar(value="Target Position: 0"),
        "steps": tk.StringVar(value="Steps: 0"),
        "visit_count": tk.StringVar(value="Regions visited: 0"),
        "current_reward": tk.StringVar(value="Reward: 0.00"),
        "episode": tk.StringVar(value=f"Episode: 0/{num_episodes}"),
        "epsilon": tk.StringVar(value=f"ε: {epsilon:.3f}"),
        "regions": tk.StringVar(value="Regions: 0"),
        "phase": tk.StringVar(value="Phase: Initial"),
        "range": tk.StringVar(value="Visible Range: [-200, 200]"),
        "test_round": tk.StringVar(value=f"Test Round: 1/{num_test_rounds}"),
    }

    # Create UI components
    info_frame = create_info_frame(root, variables)
    params_frame = create_params_frame(root, variables)
    canvas = create_canvas(root, canvas_width, canvas_height)
    create_legend(root)

    # Create visualizer
    visualizer = EnvironmentVisualizer(canvas, canvas_width, canvas_height)

    # Function to start a new test round
    def start_next_test_round():
        if current_test_round[0] >= num_test_rounds:
            # All test rounds complete, calculate the average competitive ratio
            avg_ratio = sum(competitive_ratios) / len(competitive_ratios)
            print("\n" + "=" * 80)
            print(f"COMPLETED {num_test_rounds} TEST ROUNDS")
            print(
                f"Individual competitive ratios: {[round(r, 2) for r in competitive_ratios]}"
            )
            print(f"Average competitive ratio: {avg_ratio:.2f}")
            print("=" * 80 + "\n")

            # Clear canvas text messages
            for item in canvas.find_withtag("result_text"):
                canvas.delete(item)

            # Update status variable with the result
            variables["status"].set(
                f"Test complete - Average competitive ratio: {avg_ratio:.2f}"
            )

            # Display final summary on canvas
            canvas.create_text(
                visualizer.canvas_width // 2,
                visualizer.y_position - 120,
                text=f"TEST SUMMARY",
                font=("Arial", 18, "bold"),
                fill="blue",
                tags="result_text",
            )

            canvas.create_text(
                visualizer.canvas_width // 2,
                visualizer.y_position - 80,
                text=f"Completed {num_test_rounds} test rounds",
                font=("Arial", 14),
                fill="black",
                tags="result_text",
            )

            canvas.create_text(
                visualizer.canvas_width // 2,
                visualizer.y_position - 40,
                text=f"Average competitive ratio: {avg_ratio:.2f}",
                font=("Arial", 16, "bold"),
                fill="green" if avg_ratio < 3 else "orange" if avg_ratio < 5 else "red",
                tags="result_text",
            )

            # List individual ratios
            y_offset = visualizer.y_position + 20
            for i, ratio in enumerate(competitive_ratios):
                canvas.create_text(
                    visualizer.canvas_width // 2,
                    y_offset,
                    text=f"Round {i+1}: {ratio:.2f}",
                    font=("Arial", 12),
                    fill="black",
                    tags="result_text",
                )
                y_offset += 25

            # Close the window after some time
            root.after(15000, root.destroy)
            return

        # Reset for the next test round
        current_test_round[0] += 1
        variables["test_round"].set(
            f"Test Round: {current_test_round[0]}/{num_test_rounds}"
        )

        # Reset the environment
        env.reset()

        # Clear all canvas text and markers from previous rounds
        for item in canvas.find_all():
            # Keep only the grid, agent/target markers, labels, and path line
            if (
                item != visualizer.agent_marker
                and item != visualizer.agent_label
                and item != visualizer.target_marker
                and item != visualizer.target_label
                and item != visualizer.path_line
            ):
                canvas.delete(item)

        # Reset visualizer state
        visualizer.current_min_visible = -200
        visualizer.current_max_visible = 200
        visualizer.last_search_phase = -1

        # Update grid with reset range
        visualizer.update_grid()
        visualizer.reset_path()
        visualizer.clear_markers()

        # Remove any sight range indicators
        # if visualizer.sight_range_indicator:
        #    canvas.delete(visualizer.sight_range_indicator)
        #    visualizer.sight_range_indicator = None

        # Reset agent and target markers
        agent_x = visualizer.x_to_canvas(env.current_position)
        target_x = visualizer.x_to_canvas(env.target)

        canvas.coords(
            visualizer.agent_marker,
            agent_x - visualizer.agent_radius,
            visualizer.y_position - visualizer.agent_radius,
            agent_x + visualizer.agent_radius,
            visualizer.y_position + visualizer.agent_radius,
        )
        canvas.coords(
            visualizer.agent_label,
            agent_x,
            visualizer.y_position - visualizer.agent_radius - 10,
        )

        canvas.coords(
            visualizer.target_marker,
            target_x - visualizer.target_radius,
            visualizer.y_position - visualizer.target_radius,
            target_x + visualizer.target_radius,
            visualizer.y_position + visualizer.target_radius,
        )
        canvas.coords(
            visualizer.target_label,
            target_x,
            visualizer.y_position - visualizer.target_radius - 10,
        )

        # Reset visuals
        visualizer.canvas.itemconfig(visualizer.agent_marker, fill="red")
        visualizer.canvas.itemconfig(
            visualizer.target_marker, fill="green", outline="black"
        )
        visualizer.canvas.itemconfig(visualizer.target_label, fill="black")

        # Update variables
        variables["current_pos"].set(f"Agent Position: {env.current_position}")
        variables["target_pos"].set(f"Target Position: {env.target}")
        variables["steps"].set(f"Steps: 0")
        variables["visit_count"].set(f"Regions visited: 0")
        variables["current_reward"].set(f"Reward: 0.00")
        variables["phase"].set(f"Phase: Initial")
        variables["range"].set(f"Visible Range: [-200, 200]")
        variables["status"].set(f"Starting Test Round {current_test_round[0]}")

        # Start the next test round
        update_step_for_multiple_rounds(
            env, Q, visualizer, variables, root, delay, 0, max_steps, current_mode=mode
        )

    # Modified update_step for multiple rounds
    def update_step_for_multiple_rounds(
        env, Q, visualizer, variables, root, delay, steps, max_steps, current_mode=mode
    ):
        if steps >= max_steps:
            print("Reached maximum step limit.")
            # Calculate and store the competitive ratio
            ratio = env.calculate_custom_competitive_ratio()
            competitive_ratios.append(ratio)
            env.print_competitive_ratio()

            # Remove any previous result messages
            for item in canvas.find_withtag("result_text"):
                canvas.delete(item)

            # Create status message on the canvas
            canvas.create_text(
                visualizer.canvas_width // 2,
                visualizer.y_position - 60,
                text=f"Maximum steps reached: {steps}",
                font=("Arial", 16),
                fill="orange",
                tags="result_text",
            )

            # Schedule the next test round with shorter delay
            root.after(500, start_next_test_round)
            return steps

        # Get current observation and convert to state tuple
        current_obs = env._get_observation()
        current_state = observation_to_state(current_obs)

        # Initialize info for first step
        if steps == 0:
            info = {"search_phase": env.search_phase}

        # Select action with highest Q-value for current state - pure exploitation for testing
        action = max(Q[current_state], key=Q[current_state].get)

        next_obs, reward, done, truncated, info = env.step(action)
        steps += 1

        # Check if visible range needs updating
        range_changed = visualizer.update_visible_range(
            env.current_position, env.target
        )
        if range_changed:
            # Update grid and target marker
            visualizer.update_grid()

        # Update the agent and target markers
        agent_x = visualizer.update_agent_position(env.current_position)
        target_x = visualizer.update_target_position(env.target)

        # Update sight range visualization
        # visualizer.update_sight_range(agent_x, target_x)

        # Update phases based on search phase
        if info.get("search_phase", 0) != visualizer.last_search_phase:
            visualizer.last_search_phase = info.get("search_phase", 0)
            visualizer.add_phase_marker(agent_x, visualizer.last_search_phase)

        # Update info panel with current status and reward breakdown
        visualizer.update_info_panel(
            variables,
            steps=steps,
            reward=reward,
            search_phase=info.get("search_phase", 0),
            target_found=env.target_found,
            reward_components=info.get("reward_components", {}),
            action=action,
        )

        # Update info labels
        variables["current_pos"].set(f"Agent Position: {env.current_position}")
        variables["target_pos"].set(f"Target Position: {env.target}")
        variables["range"].set(
            f"Visible Range: [{visualizer.current_min_visible}, {visualizer.current_max_visible}]"
        )

        # Update region visitation count
        regions_visited_count = len(env.regions_visited)
        variables["visit_count"].set(f"Regions visited: {regions_visited_count}")
        variables["regions"].set(f"Regions: {regions_visited_count}")

        # Check if the agent is on the target
        if env.target_found and env.rescue_complete:
            # Change the agent marker color to indicate success
            visualizer.highlight_success(True)
            print(f"Mission complete! Agent returned to base in {steps} steps!")

            # Calculate and store the competitive ratio
            ratio = env.calculate_custom_competitive_ratio()
            competitive_ratios.append(ratio)
            env.print_competitive_ratio()

            # Remove any previous result messages
            for item in canvas.find_withtag("result_text"):
                canvas.delete(item)

            # Create a success message on the canvas
            canvas.create_text(
                visualizer.canvas_width // 2,
                visualizer.y_position - 60,
                text=f"Mission complete in {steps} steps!",
                font=("Arial", 16, "bold"),
                fill="green",
                tags="result_text",
            )

            # Schedule the next test round with shorter delay
            root.after(500, start_next_test_round)
            return steps

        if done:
            print(f"Target found at position {env.current_position} in {steps} steps!")
            visualizer.highlight_success(True)

            # Calculate and store the competitive ratio
            ratio = env.calculate_custom_competitive_ratio()
            competitive_ratios.append(ratio)
            env.print_competitive_ratio()

            # Remove any previous result messages
            for item in canvas.find_withtag("result_text"):
                canvas.delete(item)

            # Create status message
            canvas.create_text(
                visualizer.canvas_width // 2,
                visualizer.y_position - 60,
                text=f"Target found at position {env.current_position} in {steps} steps!",
                font=("Arial", 16, "bold"),
                fill="blue",
                tags="result_text",
            )

            # Schedule the next test round with shorter delay
            root.after(500, start_next_test_round)
            return steps

        if truncated:
            print(f"Maximum steps reached: {steps}")

            # Calculate and store the competitive ratio
            ratio = env.calculate_custom_competitive_ratio()
            competitive_ratios.append(ratio)
            env.print_competitive_ratio()

            # Remove any previous result messages
            for item in canvas.find_withtag("result_text"):
                canvas.delete(item)

            canvas.create_text(
                visualizer.canvas_width // 2,
                visualizer.y_position - 60,
                text=f"Maximum steps reached: {steps}",
                font=("Arial", 16),
                fill="orange",
                tags="result_text",
            )

            # Schedule the next test round with shorter delay
            root.after(500, start_next_test_round)
            return steps

        root.after(
            delay,
            lambda: update_step_for_multiple_rounds(
                env,
                Q,
                visualizer,
                variables,
                root,
                delay,
                steps,
                max_steps,
                current_mode,
            ),
        )
        return steps

    # Initial grid setup
    visualizer.update_grid()

    # Create agent and target markers
    agent_x = visualizer.x_to_canvas(env.current_position)
    target_x = visualizer.x_to_canvas(env.target)

    visualizer.agent_marker, visualizer.agent_label = create_agent_marker(
        canvas, agent_x, visualizer.y_position, visualizer.agent_radius
    )

    visualizer.target_marker, visualizer.target_label = create_target_marker(
        canvas, target_x, visualizer.y_position, visualizer.target_radius
    )

    # Create path line
    visualizer.path_line = create_path_line(canvas)

    # Update initial status display
    variables["current_pos"].set(f"Agent Position: {env.current_position}")
    variables["target_pos"].set(f"Target Position: {env.target}")
    variables["status"].set("Ready to start...")

    # Mode-specific setup
    if mode == "train":
        # Start training mode
        variables["status"].set("Starting training mode...")
        root.update()

        # Execute training loop
        train_environment(
            env=env,
            Q=Q,
            visualizer=visualizer,
            variables=variables,
            root=root,
            actions=actions,
            num_episodes=num_episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            delay=delay,
        )
    else:
        # Start testing mode
        variables["status"].set("Testing learned policy...")
        variables["test_round"].set(f"Test Round: 1/{num_test_rounds}")

        # Start the first test round
        update_step_for_multiple_rounds(
            env, Q, visualizer, variables, root, delay, 0, max_steps, current_mode=mode
        )

    # Start the main loop
    root.mainloop()


# For backward compatibility with existing code
def load_q_table_wrapped(filename="saved_weights/q_table_latest.pkl"):
    """Wrapper around load_q_table for compatibility."""
    return load_q_table(filename)
