# Q-Learning for Search and Rescue on an Infinite Line

This project implements a reinforcement learning agent using Q-learning to solve a search and rescue task on a semi-infinite line (`[0, +infinity)`). The agent learns to find a potentially moving target and return it to the base (position 0). The simulation includes a Tkinter-based UI for visualizing the agent's behavior during training and testing.

The core logic is built using the Gymnasium library for the environment definition and standard Python libraries for the Q-learning algorithm and UI.

## Project Structure

```
.
├── main.py                 # Main script to run training or testing with visualization
├── requirements.txt        # Project dependencies
├── README.md               # This file
├── .gitignore
├── gymnasium_tutorial.py   # Separate Blackjack Q-learning example (not part of main project)
├── saved_weights/          # Directory to save/load trained Q-tables (e.g., q_table_latest.pkl)
└── src/
    ├── environment.py        # Defines the InfiniteLinearSearchEnv Gymnasium environment
    ├── qlearning.py        # Implements the Q-learning agent (Q-table initialization, action selection, training loop)
    ├── ui_main.py          # Main logic for the Tkinter UI, orchestrates training/testing visualization
    ├── ui_visualization.py # Handles drawing the environment state on the canvas
    ├── ui_components.py    # Defines reusable Tkinter UI elements (frames, labels, canvas markers)
    ├── ui_utils.py         # Utility functions (e.g., saving/loading Q-tables)
    └── __pycache__/        # Python cache files
```

## Environment Description (`src/environment.py`)

The `InfiniteLinearSearchEnv` simulates:

- An agent starting at the base (position 0).
- A semi-infinite line `[0, +infinity)`.
- A single target at an unknown initial position within `target_range`.
- Optional target movement (zigzag pattern with configurable probability and speed).
- A goal to find the target and return to the base (position 0).

## Key Features

- **Semi-Infinite Space**: Exploration along the positive number line.
- **Search and Rescue Task**: Find the target, then return to base.
- **Moving Target Option**: Target can be stationary or move dynamically.
- **Structured Search Phases**: Agent learns distinct phases: initial exploration, return to base, full exploration, and rescue.
- **Q-Learning Agent (`src/qlearning.py`)**: Learns a policy using a Q-table. Handles potentially infinite state space using state quantization and a default dictionary. Includes epsilon-greedy action selection with adaptive bias and experience replay.
- **Tkinter Visualization (`src/ui_*.py`)**: Provides a real-time visual representation of the agent, target, path, and key metrics during training and testing.
- **Configurable Parameters**: Environment dynamics (target movement, max steps) and learning parameters (alpha, gamma, epsilon) can be adjusted.

## Installation

1.  **Clone the repository (if applicable):**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    _Note: `requirements.txt` includes `gymnasium`, `numpy`, and `matplotlib`._

## How to Run

The primary way to run the simulation is via `main.py`.

1.  **Train a new agent:**

    ```bash
    python main.py
    ```

    This will initialize a new Q-table and start the training process, showing the visualization. The trained Q-table will be saved to `saved_weights/q_table_latest.pkl`.

2.  **Test a pre-trained agent:**
    Ensure a saved Q-table exists (e.g., `saved_weights/q_table_latest.pkl`).
    ```bash
    python main.py --load saved_weights/q_table_latest.pkl
    # or shorthand
    python main.py -l saved_weights/q_table_latest.pkl
    ```
    This will load the specified Q-table and run multiple test episodes (default: 100) with visualization, reporting the average competitive ratio at the end.

## Environment Details (`src/environment.py`)

### Initialization Parameters

- `max_steps` (int): Maximum steps per episode.
- `target_range` (int): Max initial placement position for the target.
- `region_size` (int): Size of regions for tracking visitation (used internally).
- `move_target` (bool): Whether the target moves.
- `target_move_prob` (float): Probability of target moving per step.
- `target_speed` (int): Max distance target moves in one step.
- `seed` (Optional[int]): Random seed.

### Observation Space

The observation is a Gymnasium dictionary space containing:

- `direction` (Discrete(2)): Agent's facing direction (0: left, 1: right).
- `farthest_right_rel` (Box): Farthest position reached relative to the agent's current position. _Note: Quantized in `qlearning.py` for the Q-table state._
- `target_found` (Discrete(2)): Binary flag (0: no, 1: yes).
- `distance_to_base` (Box): Absolute distance from the agent to the base (position 0). _Note: Quantized in `qlearning.py` for the Q-table state._
- `search_phase` (Discrete(3)): Current phase (0: initial, 1: return, 2: explore).

### Action Space

The action space is discrete with three possible actions:

- `0`: Move left.
- `1`: Move right.
- `2`: Signal return to base (intended to trigger phase transition logic, although primary transitions are state-based).

_Note: The agent cannot move to a position less than 0._

### Reward Structure

The reward system encourages efficient search and rescue:

- Small penalty per step (`-0.1`).
- Rewards for progressing through search phases (moving right initially, moving left to return, exploring new territory).
- Bonus for visiting new regions.
- Large reward for finding the target (`+50.0`).
- Rewards for returning to base during the rescue phase (moving left, reducing distance).
- Large reward for completing the rescue (`+100.0`).
- Penalties for counter-productive actions (e.g., moving away from base during rescue).

_See `src/environment.py`'s `step` method for the detailed reward calculation._

### State Representation for Q-Learning (`src/qlearning.py`)

Since the position and distance can be continuous/large, the raw observation is converted into a discrete state tuple for the Q-table key using `observation_to_state`:

```python
(
    direction,
    quantized_farthest_right_rel, # Farthest right relative pos quantized
    target_found,
    quantized_base_distance,    # Distance to base quantized
    search_phase,
)
```

Quantization helps manage the size of the state space.

## Visualization UI

The Tkinter UI (`src/ui_main.py`, `src/ui_visualization.py`, `src/ui_components.py`) provides:

- A visual representation of the 1D line.
- Markers for the agent (red circle) and target (green circle).
- A line showing the agent's path.
- Indicators for search phase transitions.
- Dynamic scaling and panning of the view based on agent/target position.
- Real-time display of metrics (steps, position, reward, episode, etc.).
- Summary statistics after testing multiple rounds (average competitive ratio).

## Configuration

- **Environment Verbosity:** Modify `PRINT_*` constants at the top of `src/environment.py` to control console output detail.
- **Learning Parameters:** Adjust `alpha`, `gamma`, `epsilon`, `num_episodes` in `main.py`.
- **UI Delay:** Modify the `delay` parameter passed to `test_policy_ui` in `main.py` to speed up or slow down the visualization.
