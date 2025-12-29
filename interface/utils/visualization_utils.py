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
    
    sns.heatmap(V_grid, annot=True, fmt='.2f', cmap='RdYlGn', 
          cbar_kws={'label': 'Value'}, ax=ax, square=True)
    
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
  
  if hasattr(env, 'size'):
    size = env.size
    pi_grid = pi.reshape(size, size)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    cmap = mpl.colormaps.get_cmap('tab10').resampled(4)
    im = ax.imshow(pi_grid, cmap=cmap, vmin=0, vmax=3)
    
    arrows = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    
    for i in range(size):
      for j in range(size):
        a = pi_grid[i, j]
        ax.text(j, i, arrows[a], ha='center', va='center',
            fontsize=24, color='white', fontweight='bold')
    
    ax.set_title('Learned Policy', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    
    if from_q:
      cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
      cbar.ax.set_yticklabels(['Up', 'Right', 'Down', 'Left'])
    
    return fig
  else:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(pi)), pi, color='steelblue')
    ax.set_xlabel('State')
    ax.set_ylabel('Action')
    ax.set_title('Learned Policy')
    ax.grid(True, alpha=0.3)
    return fig

def plot_policy_from_pi(pi, env):
  """Plot policy from policy array (backward compatibility)"""
  return plot_policy(pi, env, from_q=False)

def plot_qvalues(Q, env):
  """Plot Q-values for all states"""
  
  if hasattr(env, 'size'):
    size = env.size
    n_actions = Q.shape[1]
    
    fig, axes = plt.subplots(1, n_actions, figsize=(16, 4))
    fig.suptitle('Q-Values for Each Action', fontsize=14, fontweight='bold')
    
    names = ['Up', 'Right', 'Down', 'Left']
    
    for a in range(n_actions):
      Q_grid = Q[:, a].reshape(size, size)
      ax = axes[a]
      
      sns.heatmap(Q_grid, annot=True, fmt='.2f', cmap='RdYlGn',
            ax=ax, square=True, cbar=True)
      ax.set_title(f'{names[a]}')
      ax.set_xlabel('X')
      ax.set_ylabel('Y')
    
    plt.tight_layout()
    return fig
  else:
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(Q.T, aspect='auto', cmap='RdYlGn')
    ax.set_xlabel('State')
    ax.set_ylabel('Action')
    ax.set_title('Q-Values')
    plt.colorbar(im, ax=ax)
    return fig

def plot_convergence(hist):
  """Plot convergence metrics"""
  
  fig, ax = plt.subplots(figsize=(10, 6))
  
  if 'episode_rewards' in hist:
    rewards = hist['episode_rewards']
    ax.plot(rewards, alpha=0.3, label='Episode Reward')
    
    if len(rewards) > 100:
      smooth = moving_average(rewards, 100)
      ax.plot(smooth, linewidth=2, label='MA(100)')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Convergence: Reward over Episodes')
    ax.legend()
    ax.grid(True, alpha=0.3)
  
  elif 'deltas' in hist:
    ax.plot(hist['deltas'], linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Delta')
    ax.set_title('Convergence: Value Function Delta')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
  
  return fig

def plot_state_visits(visits, env):
  """Plot state visitation frequency"""
  
  if hasattr(env, 'size'):
    size = env.size
    grid = np.array(visits).reshape(size, size)
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    sns.heatmap(grid, annot=True, fmt='d', cmap='YlOrRd',
          cbar_kws={'label': 'Visits'}, ax=ax, square=True)
    
    ax.set_title('State Visitation Frequency', fontsize=14, fontweight='bold')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    
    return fig
  else:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(visits)), visits, color='coral')
    ax.set_xlabel('State')
    ax.set_ylabel('Visit Count')
    ax.set_title('State Visitation Frequency')
    ax.grid(True, alpha=0.3)
    return fig

def moving_average(data, window):
  """Calculate moving average"""
  return np.convolve(data, np.ones(window)/window, mode='valid')

def create_comparison_plots(histories, labels):
  """Create comparison plots for multiple algorithms"""
  
  fig, axes = plt.subplots(2, 2, figsize=(14, 10))
  fig.suptitle('Algorithm Comparison', fontsize=16, fontweight='bold')
  
  colors = mpl.colormaps['tab10'](np.linspace(0, 1, len(histories)))
  
  for idx, (hist, lbl) in enumerate(zip(histories, labels)):
    color = colors[idx]
    
    if 'episode_rewards' in hist:
      ax = axes[0, 0]
      rewards = hist['episode_rewards']
      if len(rewards) > 100:
        smooth = moving_average(rewards, 100)
        ax.plot(smooth, color=color, label=lbl, linewidth=2)
      ax.set_xlabel('Episode')
      ax.set_ylabel('Avg Reward')
      ax.set_title('Reward Comparison')
      ax.legend()
      ax.grid(True, alpha=0.3)
      
      ax = axes[0, 1]
      lengths = hist['episode_lengths']
      if len(lengths) > 100:
        smooth = moving_average(lengths, 100)
        ax.plot(smooth, color=color, label=lbl, linewidth=2)
      ax.set_xlabel('Episode')
      ax.set_ylabel('Steps')
      ax.set_title('Episode Length Comparison')
      ax.legend()
      ax.grid(True, alpha=0.3)
  
  plt.tight_layout()
  return fig
