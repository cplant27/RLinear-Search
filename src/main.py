from environment import InfiniteLinearSearchEnv
from qlearning import initialize_q_table
from ui_main import load_q_table_wrapped, test_policy_ui


def main(load_weights=None):
    print("Initializing configuration parameters...")
    # Configuration parameters.
    actions = [0, 1]  # 0: left, 1: right.

    # Learning parameters
    alpha = 0.6  # Learning rate
    gamma = 0.99  # Discount factor
    epsilon = 0.2  # Exploration rate (reduced from 0.8)
    num_episodes = 1000  # Number of training episodes

    print("Initializing search and rescue environment...")
    # Initialize the environment with infinite line and moving target
    env = InfiniteLinearSearchEnv(
        max_steps=5000,  # Maximum steps per episode
        target_range=500,  # Range for initial target placement (reduced from 1000)
        region_size=50,
        move_target=True,
        target_move_prob=0.2,  # Target moves with 20% probability each step (increased from 5%)
        target_speed=3,  # Max 3 units per move
    )

    # Either load Q-table from file or train from scratch
    if load_weights:
        print(f"Loading Q-table from {load_weights}...")
        Q = load_q_table_wrapped(load_weights)
        if Q is None:
            print("Failed to load Q-table. Initializing a new one...")
            Q = initialize_q_table(env, actions)
    else:
        print("Starting training to learn search and rescue strategy...")
        # Initialize Q-table
        Q = initialize_q_table(env, actions)

        # Start Q-learning training with visualization
        test_policy_ui(
            env,
            Q,
            actions,
            mode="train",
            num_episodes=num_episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
        )

    print("Training complete. Resetting environment for testing...")
    # Reset environment for testing and launch the UI visualization.
    _, _ = env.reset()
    print(f"Testing learned policy: Target is at position {env.target}")
    test_policy_ui(env, Q, actions, delay=1)  # Faster movement in testing phase

    print("Visualization complete.")


if __name__ == "__main__":
    import argparse
    import sys

    # Create an argument parser for command-line options
    parser = argparse.ArgumentParser(
        description="Search and Rescue with Q-learning and Moving Target"
    )
    parser.add_argument("--load", "-l", type=str, help="Load Q-table from file")

    args = parser.parse_args()

    main(load_weights=args.load)
