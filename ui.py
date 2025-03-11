import tkinter as tk
from typing import Dict, List
from environment import LinearSearchEnv


def test_policy_ui(
    env: LinearSearchEnv,
    Q: Dict[int, Dict[int, float]],
    actions: List[int],
    max_steps: int = 500,
    delay: int = 300,
) -> None:
    """
    Test the learned policy using a Tkinter UI.
    A canvas displays the state space with:
      - A red circle for the agent.
      - A green circle for the target.
    The agent moves are updated every 'delay' milliseconds.
    """
    lower_bound, upper_bound = env.lower_bound, env.upper_bound
    canvas_width = 800
    canvas_height = 200
    scale = canvas_width / (upper_bound - lower_bound)
    y_position = canvas_height // 2
    agent_radius = 8
    target_radius = 8

    def state_to_x(state: int) -> int:
        return int((state - lower_bound) * scale)

    # Create the Tkinter window
    root = tk.Tk()
    root.title("Q-learning Linear Search Visualization")
    canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg="white")
    canvas.pack()
    canvas.create_line(0, y_position, canvas_width, y_position, fill="black")

    # Draw target marker (green circle)
    target_x = state_to_x(env.target)
    canvas.create_oval(
        target_x - target_radius,
        y_position - target_radius,
        target_x + target_radius,
        y_position + target_radius,
        fill="green",
        outline="",
    )
    # Draw agent marker (red circle)
    agent_marker = canvas.create_oval(
        state_to_x(env.position) - agent_radius,
        y_position - agent_radius,
        state_to_x(env.position) + agent_radius,
        y_position + agent_radius,
        fill="red",
        outline="",
    )

    steps = 0

    def update_step():
        nonlocal steps, env
        if steps >= max_steps:
            print("Reached maximum step limit.")
            return

        current_state = env.position
        action = max(Q[current_state], key=Q[current_state].get)
        next_state, reward, done = env.step(action)
        steps += 1

        new_agent_x = state_to_x(env.position)
        canvas.coords(
            agent_marker,
            new_agent_x - agent_radius,
            y_position - agent_radius,
            new_agent_x + agent_radius,
            y_position + agent_radius,
        )
        print(
            f"Step {steps}: State {current_state} -> {env.position} using action {action}"
        )

        if done:
            print(f"Target found at position {env.position} in {steps} steps!")
            return
        root.after(delay, update_step)

    root.after(delay, update_step)
    root.mainloop()
