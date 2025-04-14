import tkinter as tk
from typing import Any, Dict, List


def create_info_frame(root, variables: Dict[str, tk.StringVar]) -> tk.Frame:
    """Create the information frame with status indicators."""
    info_frame = tk.Frame(root, padx=10, pady=10)
    info_frame.pack(side=tk.TOP, fill=tk.X)

    # Create labels for position, steps, etc.
    tk.Label(info_frame, text="Status:", font=("Arial", 12, "bold")).pack(
        side=tk.LEFT, padx=5
    )
    tk.Label(info_frame, textvariable=variables["status"], font=("Arial", 12)).pack(
        side=tk.LEFT, padx=10
    )

    # Create second row for position info
    position_frame = tk.Frame(root, padx=10)
    position_frame.pack(side=tk.TOP, fill=tk.X)

    # Position information
    tk.Label(
        position_frame, textvariable=variables["current_pos"], font=("Arial", 10)
    ).pack(side=tk.LEFT, padx=10)
    tk.Label(
        position_frame, textvariable=variables["target_pos"], font=("Arial", 10)
    ).pack(side=tk.LEFT, padx=10)
    tk.Label(position_frame, textvariable=variables["steps"], font=("Arial", 10)).pack(
        side=tk.LEFT, padx=10
    )
    tk.Label(
        position_frame, textvariable=variables["visit_count"], font=("Arial", 10)
    ).pack(side=tk.LEFT, padx=10)
    tk.Label(
        position_frame, textvariable=variables["current_reward"], font=("Arial", 10)
    ).pack(side=tk.LEFT, padx=10)

    return info_frame


def create_params_frame(root, variables: Dict[str, tk.StringVar]) -> tk.Frame:
    """Create a frame for training parameters."""
    params_frame = tk.Frame(
        root, padx=10, pady=5, bg="#f0f0f0", relief=tk.RAISED, borderwidth=2
    )
    params_frame.pack(side=tk.TOP, fill=tk.X)

    # Create parameter labels
    tk.Label(
        params_frame,
        text="Training Parameters:",
        font=("Arial", 12, "bold"),
        bg="#f0f0f0",
    ).pack(side=tk.LEFT, padx=5)
    tk.Label(
        params_frame,
        textvariable=variables["episode"],
        font=("Arial", 12),
        bg="#f0f0f0",
        fg="#0066cc",
    ).pack(side=tk.LEFT, padx=10)
    tk.Label(
        params_frame,
        textvariable=variables["regions"],
        font=("Arial", 12),
        bg="#f0f0f0",
        fg="#006600",
    ).pack(side=tk.LEFT, padx=10)
    tk.Label(
        params_frame,
        textvariable=variables["epsilon"],
        font=("Arial", 12),
        bg="#f0f0f0",
        fg="#cc0000",
    ).pack(side=tk.LEFT, padx=10)

    # Add search phase indicator
    tk.Label(
        params_frame,
        textvariable=variables["phase"],
        font=("Arial", 12, "bold"),
        bg="#f0f0f0",
        fg="#9900cc",
    ).pack(side=tk.LEFT, padx=10)

    # Add range indicator if provided
    if "range" in variables:
        tk.Label(
            params_frame,
            textvariable=variables["range"],
            font=("Arial", 10),
            bg="#f0f0f0",
            fg="#666666",
        ).pack(side=tk.RIGHT, padx=10)

    return params_frame


def create_canvas(root, width, height) -> tk.Canvas:
    """Create the main canvas for visualization."""
    canvas = tk.Canvas(
        root,
        width=width,
        height=height,
        bg="white",
        bd=2,
        relief=tk.SUNKEN,
    )
    canvas.pack(padx=10, pady=10)

    return canvas


def create_agent_marker(canvas, x, y, radius, label_text="Agent") -> tuple:
    """Create an agent marker on the canvas."""
    agent_marker = canvas.create_oval(
        x - radius,
        y - radius,
        x + radius,
        y + radius,
        fill="red",
        outline="black",
        tags="agent",
    )
    agent_label = canvas.create_text(
        x, y - radius - 10, text=label_text, font=("Arial", 8), tags="agent_label"
    )

    return agent_marker, agent_label


def create_target_marker(canvas, x, y, radius, label_text="Target") -> tuple:
    """Create a target marker on the canvas."""
    target_marker = canvas.create_oval(
        x - radius,
        y - radius,
        x + radius,
        y + radius,
        fill="green",
        outline="black",
        tags="target",
    )
    target_label = canvas.create_text(
        x, y - radius - 10, text=label_text, font=("Arial", 8), tags="target_label"
    )

    return target_marker, target_label


def create_path_line(canvas, color="blue", width=2):
    """Create a line for tracking the agent's path."""
    return canvas.create_line(0, 0, 0, 0, fill=color, width=width, tags="path")


def create_grid(
    canvas, width, height, min_visible, max_visible, interval=50, y_position=None
):
    """Create a reference grid on the canvas."""
    visible_range = max_visible - min_visible
    scale = width / visible_range
    grid_items = []

    if y_position is None:
        y_position = height // 2

    # Draw vertical grid lines and labels
    for x in range(min_visible, max_visible + 1, interval):
        canvas_x = int((x - min_visible) * scale)

        # Grid line
        grid_line = canvas.create_line(
            canvas_x, 0, canvas_x, height, fill="#dddddd", tags="grid"
        )
        grid_items.append(grid_line)

        # Position label
        if x % (interval * 2) == 0:  # Only label every other grid line
            grid_label = canvas.create_text(
                canvas_x, height - 5, text=str(x), font=("Arial", 8), tags="grid_label"
            )
            grid_items.append(grid_label)

    # Highlight the origin (position 0)
    origin_x = (
        int((0 - min_visible) * scale) if min_visible <= 0 <= max_visible else -100
    )
    if origin_x >= 0:
        origin_line = canvas.create_line(
            origin_x, 0, origin_x, height, fill="#ff0000", width=2, tags="origin"
        )
        grid_items.append(origin_line)

        origin_label = canvas.create_text(
            origin_x,
            5,
            text="BASE",
            font=("Arial", 8, "bold"),
            fill="red",
            tags="origin_label",
        )
        grid_items.append(origin_label)

    # Draw horizontal middle line
    middle_line = canvas.create_line(
        0, y_position, width, y_position, fill="#f0f0f0", tags="middle_line"
    )
    grid_items.append(middle_line)

    return grid_items


def create_legend(root):
    """Create a legend explaining the markers used in the visualization."""
    legend_frame = tk.Frame(root, padx=10, pady=5, bg="white")
    legend_frame.pack(side=tk.BOTTOM, fill=tk.X)

    # Create a label for the legend
    tk.Label(
        legend_frame,
        text="Legend:",
        font=("Arial", 10, "bold"),
        bg="white",
    ).pack(side=tk.LEFT, padx=5)

    # Create marker examples with labels
    markers = [
        ("Agent", "red", "circle"),
        ("Target", "green", "circle"),
        ("Return Phase", "blue", "circle"),
        ("Full Exploration", "green", "circle"),
        ("Path", "blue", "line"),
    ]

    for label_text, color, shape in markers:
        sample_frame = tk.Frame(legend_frame, bg="white")
        sample_frame.pack(side=tk.LEFT, padx=10)

        # Create marker sample
        if shape == "circle":
            canvas_sample = tk.Canvas(
                sample_frame, width=15, height=15, bg="white", highlightthickness=0
            )
            canvas_sample.create_oval(3, 3, 12, 12, fill=color, outline="black")
        else:  # line
            canvas_sample = tk.Canvas(
                sample_frame, width=15, height=15, bg="white", highlightthickness=0
            )
            canvas_sample.create_line(3, 7, 12, 7, fill=color, width=2)

        canvas_sample.pack(side=tk.LEFT)

        # Create label
        tk.Label(sample_frame, text=label_text, font=("Arial", 8), bg="white").pack(
            side=tk.LEFT, padx=3
        )

    return legend_frame
