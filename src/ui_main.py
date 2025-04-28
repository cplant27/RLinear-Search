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
    delay: int = 300,  # Slower movement speed for better visualization
    mode: str = "test",
    num_episodes: int = 1,
    alpha: float = 0.6,
    gamma: float = 0.99,
    epsilon: float = 0.8,
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
    """
    # Set up canvas dimensions
    canvas_width = 1200
    canvas_height = 300

    # Create the main window
    root = tk.Tk()
    root.title("Search and Rescue - Q-learning Visualization")
    root.geometry(f"{canvas_width}x{canvas_height + 200}")

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
    }

    # Create UI components
    info_frame = create_info_frame(root, variables)
    params_frame = create_params_frame(root, variables)
    canvas = create_canvas(root, canvas_width, canvas_height)
    create_legend(root)

    # Create visualizer
    visualizer = EnvironmentVisualizer(canvas, canvas_width, canvas_height)

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
        )
    else:
        # Start testing mode
        variables["status"].set("Testing learned policy...")
        steps = 0

        # Start the step loop
        update_step(env, Q, visualizer, variables, root, delay, steps, max_steps)

    # Start the main loop
    root.mainloop()


# For backward compatibility with existing code
def load_q_table_wrapped(filename="saved_weights/q_table_latest.pkl"):
    """Wrapper around load_q_table for compatibility."""
    return load_q_table(filename)
