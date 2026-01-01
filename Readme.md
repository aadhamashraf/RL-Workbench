# Reinforcement Learning Playground

<div align="center">

**An Interactive Streamlit Application for Visualizing and Understanding Reinforcement Learning Algorithms**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)](https://streamlit.io/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-0.29.1-green.svg)](https://gymnasium.farama.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Algorithms](#-algorithms) • [Environments](#-environments) • [Project Structure](#-project-structure)

</div>

---

## Overview

The **RL Playground** is a comprehensive educational tool designed to help students, researchers, and enthusiasts understand reinforcement learning algorithms through interactive visualization. Train agents, compare algorithms, and watch them learn in real-time across 28+ diverse environments.

## 🎥 Demo

<p align="center">
  <img src="interface/RL Bonus Visualizer.gif" width="800" alt="RL Playground Demo"/>
</p>

###  Key Highlights

-  **7 Classic RL Algorithms** - From Dynamic Programming to Temporal Difference methods
-  **28+ Environments** - GridWorlds, Mazes, Classic Control, and Gymnasium environments
-  **Rich Visualizations** - Training progress, convergence curves, and policy animations
-  **Interactive Training** - Adjust hyperparameters and see results instantly
-  **Performance Analytics** - Compare algorithms and track training history
-  **Video Generation** - Automatic GIF/video creation of learned policies

---

##  Features

###  Educational Focus
- **Step-by-step visualization** of how agents learn
- **Dual video system**: Training progress + Final learned policy
- **Convergence metrics** for understanding algorithm behavior
- **Parameter tuning** to explore hyperparameter effects

###  Research & Experimentation
- **Training history** storage for all runs
- **Algorithm comparison** on identical environments
- **Performance metrics** including rewards, steps, and convergence
- **Export capabilities** for videos and training data

###  User Experience
- **Modern UI** with gradient themes and smooth animations
- **Responsive design** that works on desktop and mobile
- **Three main tabs**: Training, Inference, and Design/Analysis
- **Real-time feedback** during training

---

##  Installation

### Prerequisites

- **Python 3.10** or higher
- **Anaconda** (recommended) or pip

### Option 1: Using Conda (Recommended)

```bash
# Download and install Anaconda
# Visit: https://www.anaconda.com/products/distribution

# Verify Conda installation
conda --version

# Create a new conda environment
conda create -n rl python=3.10 -y

# Activate the environment
conda activate rl

# Install Gymnasium and dependencies
conda install -c conda-forge gymnasium box2d-py pygame -y

# Clone the repository
git clone <repository-url>
cd Bonus

# Install Python dependencies
pip install -r requirements.txt
```

### Option 2: Using pip

```bash
# Clone the repository
git clone <repository-url>
cd Bonus

# Create a virtual environment
python -m venv venv

# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Check Gymnasium
gymnasium --version
# Expected output: gymnasium 0.x.x

# Check Streamlit
streamlit --version
# Expected output: Streamlit, version 1.29.0
```

---

##  Usage

### Starting the Application

```bash
# Make sure your environment is activated
conda activate rl  # or source venv/bin/activate

# Run the Streamlit app
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Quick Start Guide

1. **Select an Environment** (Sidebar)
   - Choose from 28+ environments grouped by category
   - View environment details and difficulty

2. **Choose an Algorithm** (Sidebar)
   - Pick from 7 RL algorithms
   - See algorithm type and features

3. **Configure Parameters** (Sidebar)
   - Adjust learning rate, discount factor, episodes, etc.
   - Parameters adapt based on selected algorithm

4. **Train Your Agent** (Training Tab)
   - Click "Start Training" button
   - Watch real-time progress and metrics
   - View training videos and convergence plots

5. **Test the Policy** (Inference Tab)
   - Run the learned policy
   - Generate test videos
   - Evaluate performance

6. **Analyze Results** (Design Tab)
   - Browse training history
   - Compare different runs
   - Filter by environment and algorithm

---

##  Algorithms

The playground implements 7 fundamental RL algorithms across 4 categories:

###  Dynamic Programming

#### Value Iteration
- **Type**: Model-based planning
- **Description**: Iteratively updates value function to find optimal policy
- **Features**: Fast convergence, requires environment model
- **Best for**: Small discrete environments with known dynamics
- **Parameters**: `gamma` (discount factor), `theta` (convergence threshold)

#### Policy Iteration
- **Type**: Model-based planning
- **Description**: Alternates between policy evaluation and improvement
- **Features**: Guaranteed convergence, full sweeps
- **Best for**: Environments where policy evaluation is efficient
- **Parameters**: `gamma`, `theta`, `max_iterations`

###  Monte Carlo Methods

#### Monte Carlo Control
- **Type**: Model-free learning
- **Description**: Learns from complete episode returns
- **Features**: Episode-based, high variance, no bootstrapping
- **Best for**: Episodic tasks with clear terminal states
- **Parameters**: `epsilon` (exploration), `gamma`, `episodes`

###  Temporal Difference Learning

#### TD(0)
- **Type**: Value prediction
- **Description**: One-step bootstrapping for state value learning
- **Features**: Low variance, biased estimates
- **Best for**: Understanding TD fundamentals (prediction only)
- **Parameters**: `alpha` (learning rate), `gamma`, `episodes`

#### n-step TD
- **Type**: Value prediction
- **Description**: Multi-step bootstrapping with adjustable lookahead
- **Features**: Tunable bias-variance tradeoff
- **Best for**: Balancing MC and TD(0) approaches
- **Parameters**: `alpha`, `gamma`, `n` (steps), `episodes`

###  Model-Free Control

#### SARSA (State-Action-Reward-State-Action)
- **Type**: On-policy TD control
- **Description**: Learns Q-values using actual actions taken
- **Features**: Conservative, safe learning, on-policy
- **Best for**: Safety-critical applications, stochastic environments
- **Parameters**: `alpha`, `gamma`, `epsilon`, `episodes`

#### Q-Learning
- **Type**: Off-policy TD control
- **Description**: Learns optimal Q-values using max action
- **Features**: Aggressive, optimal convergence, off-policy
- **Best for**: Finding optimal policies, deterministic environments
- **Parameters**: `alpha`, `gamma`, `epsilon`, `episodes`

### Algorithm Comparison

| Algorithm | Type | Model-Free | Bootstrapping | Best Use Case |
|-----------|------|------------|---------------|---------------|
| Value Iteration | DP | ❌ | ✅ | Known dynamics |
| Policy Iteration | DP | ❌ | ✅ | Small state spaces |
| Monte Carlo | MC | ✅ | ❌ | Episodic tasks |
| TD(0) | TD | ✅ | ✅ | Value prediction |
| n-step TD | TD | ✅ | ✅ | Flexible learning |
| SARSA | Control | ✅ | ✅ | Safe exploration |
| Q-Learning | Control | ✅ | ✅ | Optimal policies |

---

##  Environments

The playground features **28 environments** across 5 categories:

###  GridWorld Variations (6 environments)

Classic grid navigation with obstacles and goals.

| Environment | Size | Difficulty | Description |
|-------------|------|------------|-------------|
| `GridWorld-Small` | 5×5 | ⭐ Easy | Beginner-friendly navigation |
| `GridWorld-Sparse` | 10×10 | ⭐ Easy | Few obstacles, easier paths |
| `GridWorld` | 10×10 | ⭐⭐ Medium | Standard configuration |
| `GridWorld-Medium` | 10×10 | ⭐⭐ Medium | Balanced obstacles |
| `GridWorld-Dense` | 10×10 | ⭐⭐⭐ Hard | Many obstacles |
| `GridWorld-Large` | 15×15 | ⭐⭐⭐ Hard | Large complex grid |

**State Space**: Discrete (grid positions)  
**Action Space**: Discrete (Up, Down, Left, Right)

---

###  Gymnasium Environments (11 environments)

Standard RL benchmarks from the Gymnasium library.

#### Toy Text Environments

| Environment | Description | Difficulty | State Space | Action Space |
|-------------|-------------|------------|-------------|--------------|
| `FrozenLake` | 4×4 frozen lake navigation | ⭐ Easy | Discrete (16) | Discrete (4) |
| `FrozenLake-8x8` | 8×8 larger frozen lake | ⭐⭐ Medium | Discrete (64) | Discrete (4) |
| `FrozenLake-Slippery` | 4×4 with stochastic dynamics | ⭐⭐⭐ Hard | Discrete (16) | Discrete (4) |
| `FrozenLake-8x8-Slippery` | 8×8 with slippery surface | ⭐⭐⭐⭐ Very Hard | Discrete (64) | Discrete (4) |
| `Taxi` | Pickup and dropoff task | ⭐⭐ Medium | Discrete (500) | Discrete (6) |
| `CliffWalking` | Navigate cliff edge | ⭐⭐ Medium | Discrete (48) | Discrete (4) |
| `Blackjack` | Card game strategy | ⭐ Easy | Discrete (704) | Discrete (2) |

#### Classic Control Environments

| Environment | Description | Difficulty | State Space | Action Space |
|-------------|-------------|------------|-------------|--------------|
| `CartPole` | Balance pole on cart (200 steps) | ⭐⭐ Medium | Continuous (4D) | Discrete (2) |
| `CartPole-Long` | Extended episode limit (500 steps) | ⭐⭐ Medium | Continuous (4D) | Discrete (2) |
| `MountainCar` | Drive up hill with momentum | ⭐⭐⭐ Hard | Continuous (2D) | Discrete (3) |
| `Acrobot` | Swing two-link robot to goal | ⭐⭐⭐ Hard | Continuous (6D) | Discrete (3) |

**Note**: Continuous state spaces are automatically discretized for tabular methods.

---

###  Maze Variations (6 environments)

Procedurally generated mazes with increasing complexity.

| Environment | Size | Difficulty | Complexity |
|-------------|------|------------|------------|
| `Maze-Tiny` | 5×5 | ⭐ Easy | Simple paths |
| `Maze-Small` | 7×7 | ⭐ Easy | Few dead ends |
| `Maze` | 10×10 | ⭐⭐ Medium | Standard maze |
| `Maze-Medium` | 10×10 | ⭐⭐ Medium | Moderate complexity |
| `Maze-Large` | 15×15 | ⭐⭐⭐ Hard | Complex paths |
| `Maze-Huge` | 20×20 | ⭐⭐⭐⭐ Very Hard | Expert level |

**State Space**: Discrete (grid positions)  
**Action Space**: Discrete (Up, Down, Left, Right)  
**Features**: Random generation, guaranteed solvability

---

###  Simple Navigation (2 environments)

Minimal environments for understanding RL fundamentals.

| Environment | Description | Difficulty | States | Actions |
|-------------|-------------|------------|--------|---------|
| `Corridor` | 1D corridor navigation | ⭐ Easy | 10 | 2 (Left, Right) |
| `TwoRooms` | Navigate between rooms via door | ⭐⭐ Medium | Variable | 4 (Cardinal) |

**Best for**: Learning basics, debugging algorithms, quick experiments

---

###  Games (1 environment)

| Environment | Description | Difficulty | States | Actions |
|-------------|-------------|------------|--------|---------|
| `TicTacToe` | Classic 3×3 game vs random opponent | ⭐ Easy | 5,478 | 9 |

**Features**: Self-play learning, strategic decision-making

---


---

## 📁 Project Structure

```
Bonus/
├── app.py                      # Main Streamlit application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── rl/                         # Core RL implementation
│   ├── __init__.py            # Package initialization
│   ├── settings.py            # Algorithm and environment metadata
│   │
│   ├── core/                  # RL algorithms
│   │   ├── __init__.py
│   │   ├── dp.py              # Dynamic Programming (Value/Policy Iteration)
│   │   ├── mc.py              # Monte Carlo methods
│   │   ├── qlearning.py       # Q-Learning algorithm
│   │   ├── td.py              # Temporal Difference methods (TD(0), n-step, SARSA)
│   │   └── wrappers/          # Environment wrappers and utilities
│   │
│   └── envs/                  # Environment implementations
│       ├── __init__.py
│       ├── base.py            # Base environment interface
│       ├── factory.py         # Environment factory
│       ├── gridworld.py       # GridWorld environments
│       ├── maze.py            # Maze environments
│       ├── corridor.py        # Corridor environment
│       ├── two_rooms.py       # TwoRooms environment
│       ├── tictactoe.py       # TicTacToe environment
│       └── gym_wrapper.py     # Gymnasium environment wrapper
│
├── interface/                 # Streamlit UI components
│   ├── __init__.py
│   ├── page_config.py         # Page styling and configuration
│   ├── sidebar.py             # Sidebar controls
│   ├── main_content.py        # Main content area
│   │
│   ├── tabs/                  # Tab implementations
│   │   ├── __init__.py
│   │   ├── training_tab.py    # Training interface
│   │   ├── inference_tab.py   # Inference/testing interface
│   │   └── design_tab.py      # Analysis and history
│   │
│   └── utils/                 # UI utilities
│       ├── video_generator.py # GIF/video generation
│       └── visualizations.py  # Plotting and charts
│
├── data/                      # Training data storage
│   └── training_history.json  # Saved training runs
│
├── outputs/                   # Generated videos and visualizations
│   ├── training_videos/       # Training progress GIFs
│   └── inference_videos/      # Learned policy GIFs
│
```
---

### Key Components

#### `rl/core/` - Algorithm Implementations
- **`dp.py`**: Value Iteration and Policy Iteration with convergence tracking
- **`mc.py`**: Monte Carlo control with epsilon-greedy exploration
- **`qlearning.py`**: Q-Learning with optimized updates
- **`td.py`**: TD(0), n-step TD, and SARSA implementations

#### `rl/envs/` - Environment Suite
- **Custom environments**: GridWorld, Maze, Corridor, TwoRooms, TicTacToe
- **Gymnasium wrapper**: Seamless integration with Gymnasium environments
- **Factory pattern**: Easy environment creation and configuration

#### `interface/` - User Interface
- **Modern design**: Gradient themes, smooth animations, responsive layout
- **Three-tab structure**: Training, Inference, Design/Analysis
- **Real-time updates**: Live training metrics and visualizations

---

##  User Interface

### Training Tab
- **Environment selection** with grouped dropdown
- **Algorithm selection** with metadata display
- **Parameter configuration** with dynamic controls
- **Training button** with progress tracking
- **Real-time metrics**: Episodes, rewards, convergence
- **Dual video display**: Training progress + Final policy
- **Convergence plots**: Rewards and delta over time

### Inference Tab
- **Policy testing** on trained agents
- **Episode configuration** for testing
- **Performance metrics**: Average reward, success rate
- **Video generation** of test episodes
- **Comparison** with training performance

### Design Tab
- **Training history** browser
- **Filter by environment** and algorithm
- **Run comparison** side-by-side
- **Performance analytics**: Charts and statistics
- **Export capabilities**: Download videos and data

---

##  Configuration

### Algorithm Parameters

Each algorithm has specific parameters that can be tuned:

```python
# Dynamic Programming
gamma = 0.99          # Discount factor
theta = 1e-6          # Convergence threshold
max_iterations = 1000 # Maximum iterations

# Monte Carlo
epsilon = 0.1         # Exploration rate
gamma = 0.99          # Discount factor
episodes = 1000       # Number of episodes

# Temporal Difference
alpha = 0.1           # Learning rate
gamma = 0.99          # Discount factor
epsilon = 0.1         # Exploration rate
episodes = 1000       # Number of episodes
n = 5                 # Steps (for n-step TD)

# Q-Learning / SARSA
alpha = 0.1           # Learning rate
gamma = 0.99          # Discount factor
epsilon = 0.1         # Exploration rate
episodes = 1000       # Number of episodes
```

### Environment Configuration

Environments can be configured in `rl/settings.py`:

```python
ENVIRONMENTS = {
    'GridWorld': {
        'description': 'Reach goal while avoiding obstacles (10x10).',
        'type': 'Custom',
        'difficulty': 'Medium',
        'state_space': 'Discrete',
        'action_space': 'Discrete',
        'env_id': None
    },
    # ... more environments
}
```

---

##  Performance Tips

### For Faster Training
1. **Start small**: Use smaller environments (GridWorld-Small, Maze-Tiny)
2. **Reduce episodes**: Lower episode count for initial experiments
3. **Use simpler algorithms**: Q-Learning and SARSA are faster than Monte Carlo
4. **Adjust convergence**: Increase `theta` for faster (but less precise) convergence

### For Better Results
1. **Tune learning rate**: Try values between 0.01 and 0.5
2. **Adjust exploration**: Balance epsilon between 0.05 and 0.3
3. **Increase episodes**: More episodes = better convergence
4. **Use appropriate algorithm**: Match algorithm to environment characteristics

### For Continuous Environments
- Automatic discretization is applied
- May require more episodes to converge
- Consider adjusting bin sizes in `gym_wrapper.py`

---

##  Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'gymnasium'`
```bash
# Solution: Install gymnasium
conda install -c conda-forge gymnasium
# or
pip install gymnasium
```

**Issue**: Video generation fails
```bash
# Solution: Install video dependencies
pip install imageio imageio-ffmpeg
```

**Issue**: Pygame display errors
```bash
# Solution: Install pygame properly
conda install -c conda-forge pygame
# or
pip install pygame
```

**Issue**: Training takes too long
- Reduce number of episodes
- Use smaller environment
- Try faster algorithm (Q-Learning instead of Monte Carlo)

**Issue**: Agent not learning
- Increase learning rate (alpha)
- Adjust exploration rate (epsilon)
- Increase number of episodes
- Check environment rewards

---

##  Contributing

Contributions are welcome! Here are some ways to contribute:

### Adding New Environments
1. Create environment class in `rl/envs/`
2. Inherit from `BaseEnvironment`
3. Implement required methods
4. Add to `ENVIRONMENTS` in `settings.py`
5. Test with `test_all_envs.py`

### Adding New Algorithms
1. Create algorithm file in `rl/core/`
2. Follow existing algorithm structure
3. Add to `ALGORITHMS` in `settings.py`
4. Update UI parameter controls
5. Add documentation

### Improving UI
1. Modify components in `interface/`
2. Follow existing design patterns
3. Test responsiveness
4. Update documentation

---

##  Learning Resources

### Reinforcement Learning
- [Sutton & Barto - RL: An Introduction](http://incompleteideas.net/book/the-book-2nd.html)
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [David Silver's RL Course](https://www.davidsilver.uk/teaching/)

### Gymnasium
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Gymnasium Environments](https://gymnasium.farama.org/environments/)

### Streamlit
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)

---

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

##  Acknowledgments

- **Gymnasium** team for the excellent RL environment library
- **Streamlit** for the amazing web framework
- **Sutton & Barto** for the foundational RL textbook
---


<div align="center">

⭐ Star this repo if you find it helpful! ⭐

</div>
