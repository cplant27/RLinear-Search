import os
import pickle
import random
import time
from collections import defaultdict
from typing import Any, Dict, List, Tuple


def print_info(message, category="INFO", step=None):
    """Print formatted diagnostic information."""
    step_str = f"[Step {step}] " if step is not None else ""
    print(f"[{category}] {step_str}{message}")


def save_q_table(Q, episode, directory="saved_weights"):
    """Save Q-table to a file using pickle."""
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # Save to file with episode number
    filename = os.path.join(directory, f"q_table_episode_{episode}.pkl")
    with open(filename, "wb") as f:
        pickle.dump(dict(Q), f)

    # Also save as latest
    latest_file = os.path.join(directory, "q_table_latest.pkl")
    with open(latest_file, "wb") as f:
        pickle.dump(dict(Q), f)

    if episode % 100 == 0:  # Only print occasionally to avoid spam
        print(f"Saved Q-table after episode {episode} to {filename}")


def load_q_table(filename="saved_weights/q_table_latest.pkl"):
    """Load Q-table from a file using pickle."""
    if not os.path.exists(filename):
        print(f"Warning: Q-table file {filename} not found")
        return None

    with open(filename, "rb") as f:
        loaded_dict = pickle.load(f)

    # Convert back to defaultdict
    def default_q_values():
        return {0: 0.0, 1: 0.0}  # Default actions

    Q = defaultdict(default_q_values)
    Q.update(loaded_dict)
    print(f"Loaded Q-table from {filename}")
    return Q
