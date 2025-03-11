from environment import LinearSearchEnv
from qlearning import initialize_q_table, train_q_learning
from ui import test_policy_ui


def main():
    # Configuration parameters
    lower_bound, upper_bound = -100, 100
    actions = [0, 1, 2]  # 0: left, 1: right, 2: search
    states = list(range(lower_bound, upper_bound + 1))
    alpha = 0.1  # Learning rate
    gamma = 0.95  # Discount factor
    epsilon = 0.2  # Exploration rate
    num_episodes = 1000

    # Initialize the environment and Q-table
    env = LinearSearchEnv(lower_bound, upper_bound)
    Q = initialize_q_table(states, actions)

    # Train the Q-learning agent
    train_q_learning(env, Q, states, actions, alpha, gamma, epsilon, num_episodes)

    # Reset environment for testing and run the UI visualization
    env.reset()
    print(f"Testing learned policy: Target is at position {env.target}")
    test_policy_ui(env, Q, actions)


if __name__ == "__main__":
    main()
    main()
