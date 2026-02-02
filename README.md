# Q-Learning Grid World

A simple example of reinforcement learning using Q-learning to navigate a grid.

## What is Reinforcement Learning?

Imagine teaching a dog to fetch a ball. You don't explain the rules - instead, the dog tries different things and learns from the results:
- Fetch the ball → gets a treat (positive reward)
- Run away → no treat (no reward)

Over time, the dog figures out: "If I fetch the ball, I get a treat!"

**Reinforcement Learning (RL)** works the same way. An **agent** (like a robot or game character) learns by:
1. Trying actions in an **environment**
2. Getting **rewards** (positive or negative) based on what happens
3. Learning which actions lead to the best outcomes

## What is Q-Learning?

Q-learning is a specific RL method where the agent builds a "cheat sheet" of how good each action is in each situation.

**The Q-Table**: Think of it as a lookup table:
- Rows = all possible situations (states)
- Columns = all possible actions
- Values = how good that action is in that situation

Example:
| State | Go Up | Go Down | Go Left | Go Right |
|-------|-------|---------|---------|----------|
| (0,0) | -2.5  | 3.1     | -1.0    | 2.8      |
| (0,1) | 1.2   | 4.5     | -0.5    | 3.2      |

Looking at state (0,1), "Go Down" has the highest value (4.5), so that's the best action.

## The Grid World

```
+---+---+---+---+---+
| S |   |   |   |   |    S = Start (where the agent begins)
+---+---+---+---+---+
|   | X |   | X |   |    X = Wall (can't walk through)
+---+---+---+---+---+
|   |   |   |   |   |    G = Goal (where we want to go)
+---+---+---+---+---+
|   | X |   | X |   |
+---+---+---+---+---+
|   |   |   |   | G |
+---+---+---+---+---+
```

**Goal**: The agent starts at S (top-left) and must learn to reach G (bottom-right) while avoiding walls.

## How the Agent Learns

### The Learning Cycle

```
┌─────────────────────────────────────────────────────┐
│  1. Look at current position (state)                │
│                    ↓                                │
│  2. Pick an action (up/down/left/right)             │
│                    ↓                                │
│  3. Move and get reward (+10 for goal, -1 per step) │
│                    ↓                                │
│  4. Update the Q-table based on what happened       │
│                    ↓                                │
│  5. Repeat until reaching the goal                  │
└─────────────────────────────────────────────────────┘
```

### Exploration vs Exploitation

The agent faces a dilemma:
- **Exploit**: Do what it knows works best
- **Explore**: Try new things that might be better

We use **epsilon-greedy** strategy:
- With probability ε (epsilon): pick a random action (explore)
- Otherwise: pick the best known action (exploit)

At first, ε = 1.0 (100% random). Over time, it decreases so the agent explores less and uses its knowledge more.

### The Q-Learning Formula

```
Q(state, action) = Q(state, action) + α × [reward + γ × max(Q(next_state)) - Q(state, action)]
```

In plain English:
- **α (alpha = 0.1)**: Learning rate. How much to trust new information vs old knowledge.
- **γ (gamma = 0.99)**: Discount factor. How much to care about future rewards vs immediate rewards.
- **reward**: What we got for this action (+10 for goal, -1 for each step).
- **max(Q(next_state))**: The best possible value from the next state.

## Code Structure

### `GridWorld` class (The Environment)
- Creates the 5x5 grid with walls
- `reset()`: Put the agent back at the start
- `step(action)`: Move the agent and return the reward
- `render()`: Display the grid

### `QLearningAgent` class (The Learner)
- Holds the Q-table
- `choose_action()`: Pick an action (explore or exploit)
- `learn()`: Update Q-values after each action

### `train()` function
Runs 500 episodes (attempts) where the agent tries to reach the goal, learning a little bit each time.

### `test_policy()` function
After training, test the learned policy without any random exploration.

## Running the Code

### Requirements
- Python 3.x
- NumPy

### Install and Run

```bash
# Install numpy if you don't have it
pip install numpy

# Run the program
python q_learning_gridworld.py
```

### What You'll See

1. **Training progress** - Shows how the agent improves over 500 episodes
2. **Learned policy** - A grid with arrows showing the best direction to move from each cell
3. **Test run** - Watch the trained agent navigate from start to goal

Example output:
```
Episode 100/500 | Avg Reward: -12.45 | Epsilon: 0.606 | Steps: 23
Episode 200/500 | Avg Reward: 1.23 | Epsilon: 0.367 | Steps: 9
...

LEARNED POLICY (arrows show best action per cell)
=========================
 S  →  →  ↓  ↓
 ↓  █  ↓  █  ↓
 ↓  →  →  →  ↓
 ↓  █  ↑  █  ↓
 →  →  ↑  →  G
=========================
```

The arrows show the shortest path the agent learned!

## Key Hyperparameters

| Parameter | Value | What it does |
|-----------|-------|--------------|
| `learning_rate` | 0.1 | How fast the agent learns (higher = faster but less stable) |
| `discount_factor` | 0.99 | How much future rewards matter (1 = future is as important as now) |
| `epsilon` | 1.0 → 0.01 | Exploration rate (starts high, decreases over time) |
| `episodes` | 500 | Number of training attempts |

## Try Changing Things

1. **Make learning faster**: Increase `learning_rate` to 0.5
2. **More exploration**: Slow down `epsilon_decay` (e.g., 0.999 instead of 0.995)
3. **Different rewards**: Change the step penalty from -1 to -0.1 (agent will wander more)
4. **Bigger grid**: Modify `grid_size` and add more walls

## Summary

1. The agent starts knowing nothing (Q-table is all zeros)
2. It tries random actions and learns from rewards
3. Gradually, it figures out the best path to the goal
4. After training, it can consistently navigate from start to goal
