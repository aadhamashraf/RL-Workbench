import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns


def plot_training_metrics(hist, algo):
    """Plot training metrics"""

    if 'episode_rewards' in hist:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{algo} Training Metrics', fontsize=16, fontweight='bold')

        ax = axes[0, 0]
        rewards = hist['episode_rewards']
        ax.plot(rewards, alpha=0.3, color='blue', label='Raw')
        if len(rewards) > 100:
            smooth = moving_average(rewards, 100)
            ax.plot(smooth, color='red', linewidth=2, label='MA(100)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        lengths = hist['episode_lengths']
        ax.plot(lengths, alpha=0.3, color='green')
        if len(lengths) > 100:
            smooth = moving_average(lengths, 100)
            ax.plot(smooth, color='darkgreen', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length')
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        if 'epsilons' in hist:
            ax.plot(hist['epsilons'], color='purple', linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Epsilon')
            ax.set_title('Exploration Rate')
            ax.grid(True, alpha=0.3)
        else:
            ax.set_visible(False)

        ax = axes[1, 1]
        if 'td_errors' in hist:
            errors = hist['td_errors']
            ax.plot(errors, color='orange', alpha=0.5)
            if len(errors) > 100:
                smooth = moving_average(errors, 100)
                ax.plot(smooth, color='darkorange', linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel('TD Error')
            ax.set_title('Temporal Difference Error')
            ax.grid(True, alpha=0.3)
        else:
            ax.set_visible(False)

    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'{algo} Convergence', fontsize=16, fontweight='bold')

        if 'deltas' in hist:
            ax = axes[0]
            ax.plot(hist['deltas'], color='blue', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Delta')
            ax.set_title('Value Function Convergence')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

        if 'policy_changes' in hist:
            ax = axes[1]
            ax.plot(hist['policy_changes'], color='red', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Number of Changes')
            ax.set_title('Policy Changes per Iteration')
            ax.grid(True, alpha=0.3)
        elif len(axes) > 1:
            axes[1].set_visible(False)

    plt.tight_layout()
    return fig


def plot_value_function(V, env):
    """Plot value function as heatmap"""

    if hasattr(env, 'size'):
        size = env.size
        V_grid = V.reshape(size, size)

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            V_grid, annot=True, fmt='.2f', cmap='RdYlGn',
            cbar_kws={'label': 'Value'}, ax=ax, square=True
        )

        ax.set_title('State Value Function V(s)', fontsize=14, fontweight='bold')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        return fig
    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(len(V)), V, color='steelblue')
        ax.set_xlabel('State')
        ax.set_ylabel('Value')
        ax.set_title('State Value Function')
        ax.grid(True, alpha=0.3)
        return fig


def plot_policy(Q_or_pi, env, from_q=True):
    """Plot policy from Q-values or policy array"""

    pi = np.argmax(Q_or_pi, axis=1) if from_q else Q_or_pi
    n_states = len(pi)

    if hasattr(env, 'size') and n_states <= 100:
        size = env.size
        pi_grid = pi.reshape(size, size)

        fig, ax = plt.subplots(figsize=(8, 8))
        cmap = mpl.colormaps.get_cmap('tab10').resampled(4)
        im = ax.imshow(pi_grid, cmap=cmap, vmin=0, vmax=3)

        arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}
        for i in range(size):
            for j in range(size):
                ax.text(j, i, arrows[pi_grid[i, j]],
                        ha='center', va='center',
                        fontsize=24, color='white', fontweight='bold')

        ax.set_title('Learned Policy', fontsize=14, fontweight='bold')
        ax.set_xticks(range(size))
        ax.set_yticks(range(size))

        if from_q:
            cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
            cbar.ax.set_yticklabels(['Up', 'Right', 'Down', 'Left'])

        return fig
    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        action_counts = np.bincount(pi)
        ax.bar(range(len(action_counts)), action_counts)
        ax.set_xlabel('Action')
        ax.set_ylabel('Frequency')
        ax.set_title('Action Distribution')
        return fig


def plot_policy_from_pi(pi, env):
    return plot_policy(pi, env, from_q=False)


def plot_qvalues(Q, env):
    """Plot Q-values as heatmap for each action"""
    
    n_states, n_actions = Q.shape
    
    if hasattr(env, 'size') and n_states <= 100:
        size = env.size
        
        fig, axes = plt.subplots(1, n_actions, figsize=(4 * n_actions, 4))
        if n_actions == 1:
            axes = [axes]
        
        action_names = ['Up', 'Right', 'Down', 'Left'] if n_actions == 4 else [f'A{i}' for i in range(n_actions)]
        
        for action_idx in range(n_actions):
            Q_action = Q[:, action_idx].reshape(size, size)
            ax = axes[action_idx]
            
            sns.heatmap(
                Q_action, annot=True, fmt='.2f', cmap='RdYlGn',
                cbar_kws={'label': 'Q-Value'}, ax=ax, square=True
            )
            ax.set_title(f'Q-values for {action_names[action_idx]}', fontweight='bold')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
        
        plt.tight_layout()
        return fig
    else:
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(Q.T, annot=False, cmap='RdYlGn', ax=ax)
        ax.set_xlabel('State')
        ax.set_ylabel('Action')
        ax.set_title('Q-values Heatmap')
        return fig


def plot_convergence(hist):
    fig, ax = plt.subplots(figsize=(10, 6))

    if 'episode_rewards' in hist:
        rewards = hist['episode_rewards']
        ax.plot(rewards, alpha=0.3)
        if len(rewards) > 100:
            ax.plot(moving_average(rewards, 100), linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Convergence')
        ax.grid(True, alpha=0.3)

    elif 'deltas' in hist:
        ax.plot(hist['deltas'])
        ax.set_yscale('log')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Delta')
        ax.set_title('Value Convergence')
        ax.grid(True, alpha=0.3)

    return fig


def plot_state_visits(visits, env):
    if hasattr(env, 'size'):
        size = env.size
        grid = np.array(visits).reshape(size, size)
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(grid, annot=True, fmt='d', cmap='YlOrRd', square=True)
        ax.set_title('State Visitation Frequency')
        return fig
    else:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(range(len(visits)), visits)
        ax.set_title('State Visits')
        return fig


def moving_average(data, window):
    return np.convolve(data, np.ones(window) / window, mode='valid')


def create_comparison_plots(histories, labels):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Algorithm Comparison', fontsize=16, fontweight='bold')

    colors = mpl.colormaps['tab10'](np.linspace(0, 1, len(histories)))

    for idx, (hist, lbl) in enumerate(zip(histories, labels)):
        if 'episode_rewards' in hist:
            rewards = hist['episode_rewards']
            if len(rewards) > 100:
                axes[0, 0].plot(
                    moving_average(rewards, 100),
                    color=colors[idx], label=lbl
                )

    axes[0, 0].set_title('Reward Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
