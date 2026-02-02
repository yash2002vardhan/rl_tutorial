"""
Q-Learning on a Grid World - A Simple Reinforcement Learning Example

This implements the Q-learning algorithm, a model-free RL method that learns
the value of actions in states without needing a model of the environment.

Key RL Concepts Demonstrated:
- Agent: The learner that takes actions
- Environment: The grid world with states and rewards
- State: Agent's position on the grid
- Action: Move up, down, left, or right
- Reward: +10 for goal, -1 for each step (encourages shortest path)
- Policy: Derived from Q-values (take action with highest Q-value)
- Q-value: Expected cumulative reward for taking action a in state s

The Q-learning update rule:
Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]

Where:
- α (alpha): Learning rate - how much new info overrides old
- γ (gamma): Discount factor - importance of future rewards
- r: Immediate reward
- s': Next state
"""

import numpy as np
import random

# ============== ENVIRONMENT ==============
class GridWorld:
    """
    Simple grid world environment.

    Grid layout (5x5):
    +---+---+---+---+---+
    | S |   |   |   |   |   S = Start (0,0)
    +---+---+---+---+---+
    |   | X |   | X |   |   X = Wall (can't pass)
    +---+---+---+---+---+
    |   |   |   |   |   |   G = Goal (4,4)
    +---+---+---+---+---+
    |   | X |   | X |   |
    +---+---+---+---+---+
    |   |   |   |   | G |
    +---+---+---+---+---+
    """

    def __init__(self):
        self.grid_size = 5
        self.start = (0, 0)
        self.goal = (4, 4)
        self.walls = [(1, 1), (1, 3), (3, 1), (3, 3)]
        self.state = self.start

        # Actions: 0=up, 1=down, 2=left, 3=right
        self.actions = [0, 1, 2, 3]
        self.action_names = ['up', 'down', 'left', 'right']

    def reset(self):
        """Reset environment to start state."""
        self.state = self.start
        return self.state

    def step(self, action):
        """
        Take an action, return (next_state, reward, done).

        This is the core environment interface in RL.
        """
        row, col = self.state

        # Calculate next position based on action
        if action == 0:    # up
            next_state = (max(0, row - 1), col)
        elif action == 1:  # down
            next_state = (min(self.grid_size - 1, row + 1), col)
        elif action == 2:  # left
            next_state = (row, max(0, col - 1))
        elif action == 3:  # right
            next_state = (row, min(self.grid_size - 1, col + 1))

        # Check if next state is a wall - if so, stay in place
        if next_state in self.walls:
            next_state = self.state

        self.state = next_state

        # Determine reward and if episode is done
        if self.state == self.goal:
            return self.state, 10.0, True   # Big reward for reaching goal
        else:
            return self.state, -1.0, False  # Small penalty to encourage short paths

    def render(self, q_table=None):
        """Visualize the grid and optionally the learned policy."""
        symbols = {
            'empty': '.',
            'wall': '█',
            'start': 'S',
            'goal': 'G',
            'agent': 'A'
        }
        arrows = ['↑', '↓', '←', '→']

        print("\n" + "=" * 25)
        for row in range(self.grid_size):
            line = ""
            for col in range(self.grid_size):
                pos = (row, col)
                if pos == self.state:
                    line += " A "
                elif pos == self.goal:
                    line += " G "
                elif pos in self.walls:
                    line += " █ "
                elif pos == self.start:
                    line += " S "
                elif q_table is not None:
                    # Show best action direction
                    best_action = np.argmax(q_table[row, col])
                    line += f" {arrows[best_action]} "
                else:
                    line += " . "
            print(line)
        print("=" * 25)


# ============== Q-LEARNING AGENT ==============
class QLearningAgent:
    """
    Q-Learning agent that learns optimal policy through trial and error.
    """

    def __init__(self, grid_size, n_actions,
                 learning_rate=0.1,
                 discount_factor=0.99,
                 epsilon=1.0,
                 epsilon_decay=0.995,
                 epsilon_min=0.01):
        """
        Initialize Q-learning agent.

        Args:
            learning_rate (α): How much to update Q-values (0-1)
            discount_factor (γ): How much to value future rewards (0-1)
            epsilon (ε): Exploration rate for ε-greedy policy
        """
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q-table: stores Q(s,a) for all state-action pairs
        # Shape: (grid_rows, grid_cols, n_actions)
        self.q_table = np.zeros((grid_size, grid_size, n_actions))

    def choose_action(self, state, training=True):
        """
        Choose action using ε-greedy policy.

        Exploration vs Exploitation trade-off:
        - With probability ε: explore (random action)
        - With probability 1-ε: exploit (best known action)
        """
        if training and random.random() < self.epsilon:
            # Explore: choose random action
            return random.randint(0, 3)
        else:
            # Exploit: choose action with highest Q-value
            row, col = state
            return np.argmax(self.q_table[row, col])

    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-value using the Q-learning update rule.

        Q(s,a) = Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]

        This is the core of Q-learning - learning from experience.
        """
        row, col = state
        next_row, next_col = next_state

        # Current Q-value
        current_q = self.q_table[row, col, action]

        # Maximum Q-value for next state (best possible future)
        if done:
            target = reward  # No future rewards if episode ended
        else:
            target = reward + self.gamma * np.max(self.q_table[next_row, next_col])

        # Q-learning update
        self.q_table[row, col, action] += self.lr * (target - current_q)

    def decay_epsilon(self):
        """Reduce exploration rate over time."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# ============== TRAINING LOOP ==============
def train(episodes=500, render_every=100):
    """
    Train the Q-learning agent.

    This is the standard RL training loop:
    1. Reset environment
    2. For each step:
       - Choose action (ε-greedy)
       - Take action, observe reward and next state
       - Update Q-values
       - Repeat until episode ends
    """
    env = GridWorld()
    agent = QLearningAgent(
        grid_size=env.grid_size,
        n_actions=len(env.actions),
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    rewards_history = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_steps = 100  # Prevent infinite loops

        while steps < max_steps:
            # 1. Choose action
            action = agent.choose_action(state)

            # 2. Take action, observe result
            next_state, reward, done = env.step(action)

            # 3. Learn from experience
            agent.learn(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        # Decay exploration rate
        agent.decay_epsilon()
        rewards_history.append(total_reward)

        # Progress logging
        if (episode + 1) % render_every == 0:
            avg_reward = np.mean(rewards_history[-render_every:])
            print(f"Episode {episode + 1}/{episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Steps: {steps}")

    return env, agent, rewards_history


def test_policy(env, agent, num_tests=3):
    """Test the learned policy (no exploration)."""
    print("\n" + "=" * 50)
    print("TESTING LEARNED POLICY (no exploration)")
    print("=" * 50)

    for test in range(num_tests):
        print(f"\n--- Test Run {test + 1} ---")
        state = env.reset()
        env.render(agent.q_table)

        total_reward = 0
        steps = 0

        while steps < 20:
            action = agent.choose_action(state, training=False)
            next_state, reward, done = env.step(action)

            print(f"Step {steps + 1}: {env.action_names[action]} -> {next_state}")
            total_reward += reward
            state = next_state
            steps += 1

            if done:
                print(f"Goal reached in {steps} steps! Total reward: {total_reward}")
                env.render(agent.q_table)
                break
        else:
            print("Failed to reach goal within step limit")


def show_q_values(agent):
    """Display the learned Q-values."""
    print("\n" + "=" * 50)
    print("LEARNED Q-VALUES")
    print("=" * 50)

    action_names = ['↑ up', '↓ down', '← left', '→ right']

    for row in range(agent.q_table.shape[0]):
        for col in range(agent.q_table.shape[1]):
            print(f"\nState ({row},{col}):")
            for a, name in enumerate(action_names):
                q_val = agent.q_table[row, col, a]
                if q_val != 0:
                    print(f"  {name}: {q_val:.2f}")


# ============== MAIN ==============
if __name__ == "__main__":
    print("=" * 50)
    print("Q-LEARNING ON GRID WORLD")
    print("=" * 50)
    print("\nGoal: Learn to navigate from S(0,0) to G(4,4)")
    print("Walls block movement. Agent learns through trial & error.\n")

    # Train the agent
    env, agent, rewards = train(episodes=500, render_every=100)

    # Show the learned policy
    print("\n" + "=" * 50)
    print("LEARNED POLICY (arrows show best action per cell)")
    print("=" * 50)
    env.reset()
    env.render(agent.q_table)

    # Test the policy
    test_policy(env, agent, num_tests=1)

    # Optional: Show Q-values for deeper understanding
    # show_q_values(agent)
