import tkinter as tk
from typing import Any, Dict, List, Tuple

from src.environment import InfiniteLinearSearchEnv
from src.qlearning import observation_to_state
from src.ui_utils import print_info


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
        self.target_markers = []  # List of target markers
        self.target_labels = []  # List of target labels
        self.path_line = None
        self.direction_markers = []
        self.grid_items = []
        self.sensing_indicator = None  # Visual indicator for sensing

        # Tracking
        self.path_points = []
        self.current_min_visible = -200
        self.current_max_visible = 200
        self.last_search_phase = -1

        # Agent/Target dimensions
        self.agent_radius = 6
        self.target_radius = 6

        # Target colors - for differentiation
        self.target_colors = ["green", "purple", "orange", "cyan", "yellow"]

    def x_to_canvas(self, actual_position: int) -> int:
        """Convert the actual position to an x-coordinate on the canvas."""
        # Dynamic scaling based on current visible range
        visible_range = self.current_max_visible - self.current_min_visible
        scale = self.canvas_width / visible_range
        return int((actual_position - self.current_min_visible) * scale)

    def canvas_to_x(self, canvas_x: int) -> int:
        """Convert a canvas x-coordinate to actual position."""
        visible_range = self.current_max_visible - self.current_min_visible
        scale = self.canvas_width / visible_range
        return int(canvas_x / scale + self.current_min_visible)

    def update_visible_range(self, position: int, targets=None) -> bool:
        """Update the visible range based on agent and target positions."""
        # Ensure both agent and all targets are visible when possible
        min_pos = position
        max_pos = position

        if targets:
            for _, target_pos in targets:
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
                # If range is too large, center on agent and targets
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
        sensing_info=None,
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
        variables["current_reward"].set(f"Reward: {reward:.2f}")

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

        # Update sensing information if provided
        if sensing_info and "sensing_info" in variables:
            dist, direction = sensing_info
            if dist > 0:
                dir_str = (
                    "Left" if direction < 0 else "Right" if direction > 0 else "None"
                )
                variables["sensing_info"].set(
                    f"Sensing: Target {dir_str} at distance {dist:.1f}"
                )
            else:
                variables["sensing_info"].set("Sensing: No targets in range")

    def clear_markers(self):
        """Clear all direction markers."""
        for marker in self.direction_markers:
            self.canvas.delete(marker)
        self.direction_markers.clear()

        # Clear target markers
        for marker in self.target_markers:
            self.canvas.delete(marker)
        self.target_markers.clear()

        # Clear target labels
        for label in self.target_labels:
            self.canvas.delete(label)
        self.target_labels.clear()

        # Clear sensing indicator
        if self.sensing_indicator:
            self.canvas.delete(self.sensing_indicator)
            self.sensing_indicator = None

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

    def create_target_markers(self, targets, canvas):
        """Create markers for all targets."""
        self.target_markers = []
        self.target_labels = []

        for i, (target_id, target_pos) in enumerate(targets):
            color_idx = i % len(self.target_colors)
            color = self.target_colors[color_idx]

            x = self.x_to_canvas(target_pos)
            marker = canvas.create_oval(
                x - self.target_radius,
                self.y_position - self.target_radius,
                x + self.target_radius,
                self.y_position + self.target_radius,
                fill=color,
                outline="black",
                tags=f"target_{target_id}",
            )

            label = canvas.create_text(
                x,
                self.y_position - self.target_radius - 10,
                text=f"Target {target_id}",
                font=("Arial", 8),
                tags=f"target_label_{target_id}",
            )

            self.target_markers.append(marker)
            self.target_labels.append(label)

    def update_targets_positions(
        self, targets, targets_found=None, targets_rescued=None
    ):
        """Update all target markers' positions."""
        for i, (target_id, target_pos) in enumerate(targets):
            if i < len(self.target_markers):
                x = self.x_to_canvas(target_pos)

                # Update target marker position
                self.canvas.coords(
                    self.target_markers[i],
                    x - self.target_radius,
                    self.y_position - self.target_radius,
                    x + self.target_radius,
                    self.y_position + self.target_radius,
                )

                # Update target label position
                self.canvas.coords(
                    self.target_labels[i], x, self.y_position - self.target_radius - 10
                )

                # Change color based on target state
                if (
                    targets_found
                    and i < len(targets_found)
                    and targets_found[target_id]
                ):
                    if targets_rescued and targets_rescued[target_id]:
                        # Target rescued
                        self.canvas.itemconfig(
                            self.target_markers[i], fill="lightgray", outline="gray"
                        )
                    else:
                        # Target found but not rescued
                        self.canvas.itemconfig(self.target_markers[i], fill="yellow")

    def update_sensing_indicator(self, position, sensing_info):
        """Update the visual indicator for sensing nearby targets."""
        x = self.x_to_canvas(position)
        dist, direction = sensing_info

        # Remove existing indicator
        if self.sensing_indicator:
            self.canvas.delete(self.sensing_indicator)
            self.sensing_indicator = None

        # Only show indicator if target is detected
        if dist > 0:
            # Create a directional indicator
            radius = 40  # Size of sensing indicator
            angle = 0 if direction > 0 else 180  # Direction angle (right or left)

            # Create a partial circle segment to show direction
            # Use a light blue semi-transparent indicator
            start_angle = angle - 45
            end_angle = angle + 45

            # Create a circle segment
            self.sensing_indicator = self.canvas.create_arc(
                x - radius,
                self.y_position - radius,
                x + radius,
                self.y_position + radius,
                start=start_angle,
                extent=90,
                outline="blue",
                fill="lightblue",
                width=2,
                style="arc",
                tags="sensing",
            )

            # Add a line to indicate direction more clearly
            line_len = 30
            dx = line_len * direction  # Direction multiplier
            line = self.canvas.create_line(
                x,
                self.y_position,
                x + dx,
                self.y_position,
                fill="blue",
                width=2,
                arrow=tk.LAST,
                tags="sensing",
            )

            # Group both elements
            if not self.sensing_indicator:
                self.sensing_indicator = line
            else:
                # Remember both elements to delete later
                self.direction_markers.append(line)

    def highlight_success(self, found=True):
        """Highlight the agent when a target is found."""
        if found:
            self.canvas.itemconfig(self.agent_marker, fill="blue")
            # Target colors are handled in update_targets_positions
        else:
            self.canvas.itemconfig(self.agent_marker, fill="red")


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
        return steps

    # Get current observation and convert to state tuple
    current_obs = env._get_observation()
    current_state = observation_to_state(current_obs)

    # Select action with highest Q-value for current state
    action = max(Q[current_state], key=Q[current_state].get)
    next_obs, reward, done, truncated, info = env.step(action)
    steps += 1

    # Check if visible range needs updating
    range_changed = visualizer.update_visible_range(env.current_position, env.targets)
    if range_changed:
        # Update grid and target marker
        visualizer.update_grid()

    # Update the agent and target markers
    agent_x = visualizer.update_agent_position(env.current_position)
    visualizer.update_targets_positions(
        env.targets, env.targets_found, env.targets_rescued
    )

    # Update sensing indicator
    sensing_info = info.get("sensing_info", (-1, 0))
    visualizer.update_sensing_indicator(env.current_position, sensing_info)

    # Update phases based on search phase
    if info.get("search_phase", 0) != visualizer.last_search_phase:
        visualizer.last_search_phase = info.get("search_phase", 0)
        visualizer.add_phase_marker(agent_x, visualizer.last_search_phase)

    # Update info panel with current status
    visualizer.update_info_panel(
        variables,
        steps=steps,
        reward=reward,
        search_phase=info.get("search_phase", 0),
        target_found=env.target_found,
        sensing_info=sensing_info,
    )

    # Update info labels
    variables["current_pos"].set(f"Agent Position: {env.current_position}")

    # Display all target positions with status
    targets_info = ", ".join(
        [
            f"Target {id}: {pos}{' (F)' if env.targets_found[id] else ''}{' (R)' if env.targets_rescued[id] else ''}"
            for id, pos in env.targets
        ]
    )
    variables["target_pos"].set(f"Targets: {targets_info}")

    variables["range"].set(
        f"Visible Range: [{visualizer.current_min_visible}, {visualizer.current_max_visible}]"
    )

    # Update region visitation count
    regions_visited_count = len(env.regions_visited)
    variables["visit_count"].set(f"Regions visited: {regions_visited_count}")
    variables["regions"].set(f"Regions: {regions_visited_count}")

    # Check if the agent completed a rescue
    if env.rescue_complete:
        # Change the agent marker color to indicate success
        visualizer.highlight_success(True)
        rescued_count = sum(env.targets_rescued)
        if env.all_targets_rescued:
            status_msg = f"Complete success! All {rescued_count} targets rescued in {steps} steps!"
        else:
            status_msg = f"Partial success! {rescued_count} of {env.num_targets} targets rescued in {steps} steps!"

        print(status_msg)

        # Create a success message on the canvas
        visualizer.canvas.create_text(
            visualizer.canvas_width // 2,
            visualizer.y_position - 60,
            text=status_msg,
            font=("Arial", 16, "bold"),
            fill="green",
        )
        return steps

    if done:
        rescued_count = sum(env.targets_rescued)
        found_targets = [i for i, found in enumerate(env.targets_found) if found]
        print(
            f"Target(s) {found_targets} found and {rescued_count} returned to base in {steps} steps!"
        )
        visualizer.highlight_success(True)
        return steps

    if truncated:
        print(f"Maximum steps reached: {steps}")
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
