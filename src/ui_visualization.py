import tkinter as tk
from typing import Any, Dict, List, Tuple

from src.environment import InfiniteLinearSearchEnv
from src.qlearning import observation_to_state


class EnvironmentVisualizer:
    """Class to visualize the environment on a canvas."""

    def __init__(self, canvas: tk.Canvas, canvas_width: int, canvas_height: int):
        self.canvas = canvas
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.y_position = canvas_height // 2

        # Visual elements
        self.agent_marker = None
        self.agent_label = None
        self.target_marker = None
        self.target_label = None
        self.path_line = None
        self.direction_markers = []
        self.grid_items = []

        # Tracking
        self.path_points = []
        self.current_min_visible = -200
        self.current_max_visible = 200
        self.last_search_phase = -1

        # Agent/Target dimensions
        self.agent_radius = 6
        self.target_radius = 6

    def x_to_canvas(self, actual_position: int) -> int:
        """Convert the actual position to an x-coordinate on the canvas."""
        # Dynamic scaling based on current visible range
        visible_range = self.current_max_visible - self.current_min_visible
        scale = self.canvas_width / visible_range
        return int((actual_position - self.current_min_visible) * scale)

    def update_visible_range(self, position: int, target_pos: int = None) -> bool:
        """Update the visible range based on agent and target positions."""
        # Ensure both agent and target are visible when possible
        min_pos = position
        max_pos = position

        if target_pos is not None:
            min_pos = min(min_pos, target_pos)
            max_pos = max(max_pos, target_pos)

        # Add margins
        margin = 100
        min_pos -= margin
        max_pos += margin

        # Check if we need to adjust the visible range
        if min_pos < self.current_min_visible or max_pos > self.current_max_visible:
            # Expand visible range to include all points with some padding
            self.current_min_visible = min(self.current_min_visible, min_pos)
            self.current_max_visible = max(self.current_max_visible, max_pos)

            # Keep visible range reasonable
            visible_range = self.current_max_visible - self.current_min_visible
            if visible_range > 2000:
                # If range is too large, center on agent and target
                center = (min_pos + max_pos) // 2
                self.current_min_visible = center - 1000
                self.current_max_visible = center + 1000

            return True  # Range was updated

        return False  # Range unchanged

    def update_grid(self, interval=50):
        """Update the grid after a visible range change."""
        # Clear existing grid
        for item in self.grid_items:
            self.canvas.delete(item)
        self.grid_items.clear()

        # Draw new grid
        visible_range = self.current_max_visible - self.current_min_visible
        scale = self.canvas_width / visible_range

        # Draw vertical grid lines and labels
        for x in range(
            int(self.current_min_visible // interval) * interval,
            int(self.current_max_visible // interval) * interval + interval,
            interval,
        ):
            canvas_x = int((x - self.current_min_visible) * scale)

            # Grid line
            grid_line = self.canvas.create_line(
                canvas_x, 0, canvas_x, self.canvas_height, fill="#dddddd", tags="grid"
            )
            self.grid_items.append(grid_line)

            # Position label
            if x % (interval * 2) == 0:  # Only label every other grid line
                grid_label = self.canvas.create_text(
                    canvas_x,
                    self.canvas_height - 5,
                    text=str(x),
                    font=("Arial", 8),
                    tags="grid_label",
                )
                self.grid_items.append(grid_label)

        # Highlight the origin (position 0)
        origin_x = (
            int((0 - self.current_min_visible) * scale)
            if self.current_min_visible <= 0 <= self.current_max_visible
            else -100
        )
        if origin_x >= 0:
            origin_line = self.canvas.create_line(
                origin_x,
                0,
                origin_x,
                self.canvas_height,
                fill="#ff0000",
                width=2,
                tags="origin",
            )
            self.grid_items.append(origin_line)

            origin_label = self.canvas.create_text(
                origin_x,
                5,
                text="BASE",
                font=("Arial", 8, "bold"),
                fill="red",
                tags="origin_label",
            )
            self.grid_items.append(origin_label)

        # Draw horizontal middle line
        middle_line = self.canvas.create_line(
            0,
            self.y_position,
            self.canvas_width,
            self.y_position,
            fill="#f0f0f0",
            tags="middle_line",
        )
        self.grid_items.append(middle_line)

    def update_path(self, x, y):
        """Update the agent's path line."""
        if not self.path_points:
            # First point, just add it
            self.path_points.append((x, y))
        else:
            # Add new point and redraw the line
            self.path_points.append((x, y))

            # Keep only the last 1000 points to avoid performance issues
            if len(self.path_points) > 1000:
                self.path_points.pop(0)

            # Flatten the points list for canvas.coords
            flat_points = [coord for point in self.path_points for coord in point]
            self.canvas.coords(self.path_line, *flat_points)

    def add_phase_marker(self, x, phase):
        """Add a marker for a search phase change."""
        if phase == 1:  # Return to base phase
            marker = self.canvas.create_oval(
                x - 8,
                self.y_position - 20,
                x + 8,
                self.y_position - 4,
                fill="blue",
                outline="white",
            )
        elif phase == 2:  # Full exploration phase
            marker = self.canvas.create_oval(
                x - 8,
                self.y_position - 20,
                x + 8,
                self.y_position - 4,
                fill="green",
                outline="white",
            )
        else:
            return None

        self.direction_markers.append(marker)
        return marker

    def update_info_panel(
        self,
        variables: Dict[str, tk.StringVar],
        steps=0,
        reward=0.0,
        search_phase=0,
        target_found=False,
    ):
        """Update information panel with current status."""
        variables["steps"].set(f"Steps: {steps}")

        if target_found:
            status_str = "Target Found! Returning to base."
        else:
            if search_phase == 0:
                status_str = "Initial Exploration (Go Right)"
            elif search_phase == 1:
                status_str = "Return to Base (Go Left)"
            elif search_phase == 2:
                status_str = "Full Exploration (Search Right)"
            else:
                status_str = "Exploring..."

        variables["status"].set(status_str)
        variables["current_reward"].set(f"Total Reward: {reward:.2f}")

        # Update phase indicator
        if target_found:
            variables["phase"].set("Phase: Rescue")
        else:
            if search_phase == 0:
                variables["phase"].set("Phase: Initial Exploration")
            elif search_phase == 1:
                variables["phase"].set("Phase: Return to Base")
            elif search_phase == 2:
                variables["phase"].set("Phase: Full Exploration")
            else:
                variables["phase"].set("Phase: Unknown")

    def clear_markers(self):
        """Clear all direction markers."""
        for marker in self.direction_markers:
            self.canvas.delete(marker)
        self.direction_markers.clear()

    def reset_path(self):
        """Reset the path tracking."""
        self.path_points.clear()
        if self.path_line:
            self.canvas.coords(self.path_line, 0, 0, 0, 0)

    def update_agent_position(self, position: int):
        """Update the agent marker's position on the canvas."""
        x = self.x_to_canvas(position)
        self.canvas.coords(
            self.agent_marker,
            x - self.agent_radius,
            self.y_position - self.agent_radius,
            x + self.agent_radius,
            self.y_position + self.agent_radius,
        )
        self.canvas.coords(
            self.agent_label, x, self.y_position - self.agent_radius - 10
        )
        self.update_path(x, self.y_position)
        return x

    def update_target_position(self, position: int):
        """Update the target marker's position on the canvas."""
        x = self.x_to_canvas(position)
        self.canvas.coords(
            self.target_marker,
            x - self.target_radius,
            self.y_position - self.target_radius,
            x + self.target_radius,
            self.y_position + self.target_radius,
        )
        self.canvas.coords(
            self.target_label, x, self.y_position - self.target_radius - 10
        )
        return x

    def highlight_success(self, found=True):
        """Highlight the agent when the target is found."""
        if found:
            self.canvas.itemconfig(self.agent_marker, fill="blue")
            self.canvas.itemconfig(self.target_marker, fill="yellow")
        else:
            self.canvas.itemconfig(self.agent_marker, fill="red")
            self.canvas.itemconfig(self.target_marker, fill="green")


def update_step(
    env: InfiniteLinearSearchEnv,
    Q: Dict,
    visualizer: EnvironmentVisualizer,
    variables: Dict[str, tk.StringVar],
    root: tk.Tk,
    delay: int,
    steps: int,
    max_steps: int,
):
    """Update one step of the visualization."""
    if steps >= max_steps:
        print("Reached maximum step limit.")
        # Print competitive ratio when reaching max steps
        env.print_competitive_ratio()
        return steps

    # Get current observation and convert to state tuple
    current_obs = env._get_observation()
    current_state = observation_to_state(current_obs)

    # Select action with highest Q-value for current state
    action = max(Q[current_state], key=Q[current_state].get)
    next_obs, reward, done, truncated, info = env.step(action)
    steps += 1

    # Check if visible range needs updating
    range_changed = visualizer.update_visible_range(env.current_position, env.target)
    if range_changed:
        # Update grid and target marker
        visualizer.update_grid()

    # Update the agent and target markers
    agent_x = visualizer.update_agent_position(env.current_position)

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

    # Check if the agent is on the target
    if env.target_found and env.rescue_complete:
        # Change the agent marker color to indicate success
        visualizer.highlight_success(True)
        print(f"Mission complete! Agent returned to base in {steps} steps!")

        # Print competitive ratio when mission complete
        env.print_competitive_ratio()

        # Create a success message on the canvas
        visualizer.canvas.create_text(
            visualizer.canvas_width // 2,
            visualizer.y_position - 60,
            text=f"Mission complete in {steps} steps!",
            font=("Arial", 16, "bold"),
            fill="green",
        )
        return steps

    if done:
        print(f"Target found at position {env.current_position} in {steps} steps!")
        visualizer.highlight_success(True)

        # Print competitive ratio when target found
        env.print_competitive_ratio()

        return steps

    if truncated:
        print(f"Maximum steps reached: {steps}")

        # Print competitive ratio when maximum steps reached
        env.print_competitive_ratio()

        visualizer.canvas.create_text(
            visualizer.canvas_width // 2,
            visualizer.y_position - 60,
            text=f"Maximum steps reached: {steps}",
            font=("Arial", 16),
            fill="orange",
        )
        return steps

    root.after(
        delay,
        lambda: update_step(
            env, Q, visualizer, variables, root, delay, steps, max_steps
        ),
    )
    return steps
