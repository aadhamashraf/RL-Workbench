import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from interface.utils.visualization_utils import plot_training_metrics
import plotly.graph_objects as go
import plotly.express as px

def render_design_tab():
    """Render the design tab for exploring training history"""
    
    st.markdown("## Training History & Analysis")
    st.markdown("Explore all training runs, compare algorithms, and analyze hyperparameter effects")
    
    if 'training_runs' not in st.session_state:
        st.session_state.training_runs = []
    
    if not st.session_state.training_runs:
        st.info("**No training runs yet!** Train an agent in the Training tab to see results here.")
        return
    
    st.markdown("---")
    view_mode = st.radio(
        "Select Analysis Mode:",
        ["All Training Runs", "Environment Analysis", "Algorithm Deep Dive", "Compare Runs"],
        horizontal=True
    )
    st.markdown("---")
    
    views = {
        "All Training Runs": render_all_runs_view,
        "Environment Analysis": render_environment_analysis_view,
        "Algorithm Deep Dive": render_algorithm_analysis_view,
        "Compare Runs": render_comparison_view
    }
    views[view_mode]()


def render_all_runs_view():
    """Display all training runs with filtering and sorting"""
    
    st.markdown("### All Training Runs")
    runs = st.session_state.training_runs
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        envs = sorted({r['environment'] for r in runs})
        env_filter = st.multiselect("Filter by Environment", envs, envs)
    
    with col2:
        algos = sorted({r['algorithm'] for r in runs})
        algo_filter = st.multiselect("Filter by Algorithm", algos, algos)
    
    with col3:
        sort_by = st.selectbox("Sort by", [
            "Timestamp (Newest)", "Timestamp (Oldest)", 
            "Avg Reward (High)", "Avg Reward (Low)", "Training Time"
        ])
    
    filtered = [r for r in runs if r['environment'] in env_filter and r['algorithm'] in algo_filter]
    
    sort_keys = {
        "Timestamp (Newest)": lambda x: -x['timestamp'],
        "Timestamp (Oldest)": lambda x: x['timestamp'],
        "Avg Reward (High)": lambda x: -(x.get('avg_reward') or -float('inf')),
        "Avg Reward (Low)": lambda x: x.get('avg_reward') or float('inf'),
        "Training Time": lambda x: x.get('training_time', 0)
    }
    filtered.sort(key=sort_keys[sort_by])
    
    st.markdown(f"**Showing {len(filtered)} of {len(runs)} runs**")
    
    if not filtered:
        st.warning("No runs match the selected filters.")
        return
    
    for idx, run in enumerate(filtered):
        render_run_card(run, idx)
        st.markdown("---")


def render_run_card(run, idx):
    """Render a single training run card"""
    
    timestamp = datetime.fromtimestamp(run['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
    
    with st.expander(f"Run #{run.get('run_id', idx+1)}: {run['algorithm']} on {run['environment']} - {timestamp}"):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Environment", run['environment'])
        with col2:
            st.metric("Algorithm", run['algorithm'])
        with col3:
            metric_val = f"{run['avg_reward']:.2f}" if run.get('avg_reward') else run.get('iterations', 'N/A')
            metric_label = "Avg Reward" if run.get('avg_reward') else "Iterations"
            st.metric(metric_label, metric_val)
        with col4:
            st.metric("Training Time", f"{run.get('training_time', 0):.2f}s")
        
        st.markdown("**Hyperparameters:**")
        cols = st.columns(min(len(run['params']), 4))
        for i, (key, val) in enumerate(run['params'].items()):
            with cols[i % 4]:
                st.write(f"**{key}:** {val:.4f}" if isinstance(val, float) else f"**{key}:** {val}")
        
        st.markdown("---")
        
        video_col1, video_col2 = st.columns(2)
        
        with video_col1:
            st.markdown("#### Training Progress")
            if run.get('training_video_path') and os.path.exists(run['training_video_path']):
                st.video(run['training_video_path'], format='video/mp4')
                with open(run['training_video_path'], 'rb') as f:
                    st.download_button(
                        "Download Training Video",
                        f.read(),
                        os.path.basename(run['training_video_path']),
                        "video/mp4",
                        key=f"dl_train_{run.get('run_id', idx)}"
                    )
            else:
                st.info("Training video not available")
        
        with video_col2:
            st.markdown("#### Final Policy")
            if run.get('inference_video_path') and os.path.exists(run['inference_video_path']):
                st.video(run['inference_video_path'], format='video/mp4')
                with open(run['inference_video_path'], 'rb') as f:
                    st.download_button(
                        "Download Inference Video",
                        f.read(),
                        os.path.basename(run['inference_video_path']),
                        "video/mp4",
                        key=f"dl_infer_{run.get('run_id', idx)}"
                    )
            else:
                st.info("Inference video not available")
        
        if run.get('history'):
            st.markdown("#### Training Metrics")
            fig = plot_training_metrics(run['history'], run['algorithm'])
            st.pyplot(fig)


def render_environment_analysis_view():
    """Freeze an environment and compare algorithms"""
    
    st.markdown("### Environment Analysis")
    st.markdown("Compare how different algorithms perform on a specific environment")
    
    runs = st.session_state.training_runs
    envs = sorted({r['environment'] for r in runs})
    
    env = st.selectbox("Select Environment", envs)
    env_runs = [r for r in runs if r['environment'] == env]
    
    st.markdown(f"**Found {len(env_runs)} runs for {env}**")
    
    algos = sorted({r['algorithm'] for r in env_runs})
    
    st.markdown("---")
    
    mode = st.radio(
        "Display Mode:",
        ["Show Best Configuration Only", "Show All Runs", "Show All with Best Highlighted"]
    )
    
    st.markdown("---")
    st.markdown("### Algorithm Performance Comparison")
    
    perf = []
    best_per_algo = {}
    
    for algo in algos:
        algo_runs = [r for r in env_runs if r['algorithm'] == algo]
        reward_runs = [r for r in algo_runs if r.get('avg_reward')]
        
        if reward_runs:
            best = max(reward_runs, key=lambda x: x['avg_reward'])
            best_per_algo[algo] = best
            perf.append({
                'Algorithm': algo,
                'Best Avg Reward': f"{best['avg_reward']:.2f}",
                'Best Training Time': f"{best.get('training_time', 0):.2f}s",
                'Total Runs': len(algo_runs),
                'Run ID': best.get('run_id', 'N/A')
            })
        else:
            best = algo_runs[0]
            best_per_algo[algo] = best
            perf.append({
                'Algorithm': algo,
                'Best Avg Reward': 'N/A (Planning)',
                'Best Training Time': f"{best.get('training_time', 0):.2f}s",
                'Total Runs': len(algo_runs),
                'Run ID': best.get('run_id', 'N/A')
            })
    
    if perf:
        st.dataframe(pd.DataFrame(perf), use_container_width=True)
    
    reward_runs = [r for r in best_per_algo.values() 
                   if r.get('history', {}).get('episode_rewards')]
    
    if reward_runs:
        st.markdown("### Best Performance Comparison")
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        
        for idx, run in enumerate(reward_runs):
            rewards = run['history']['episode_rewards']
            episodes = list(range(1, len(rewards) + 1))
            
            window = min(50, len(rewards) // 10) or 1
            avg = pd.Series(rewards).rolling(window, min_periods=1).mean()
            
            label = f"{run['algorithm']} (Run #{run.get('run_id', 'N/A')})"
            
            fig.add_trace(go.Scatter(
                x=episodes, y=rewards, mode='lines', name=label,
                line=dict(color=colors[idx % len(colors)], width=1),
                opacity=0.3, showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=episodes, y=avg, mode='lines', name=label,
                line=dict(color=colors[idx % len(colors)], width=3)
            ))
        
        fig.update_layout(
            title=f"Algorithm Comparison on {env}",
            xaxis_title="Episode", yaxis_title="Reward",
            hovermode='x unified', height=600, template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("### Detailed Run Information")
    
    if mode == "Show Best Configuration Only":
        st.markdown("**Showing only the best configuration for each algorithm**")
        for algo, run in best_per_algo.items():
            st.markdown(f"#### Best {algo} Configuration")
            render_run_card(run, 0)
            st.markdown("---")
    
    elif mode == "Show All Runs":
        st.markdown("**Showing all runs**")
        for idx, run in enumerate(env_runs):
            render_run_card(run, idx)
            if idx < len(env_runs) - 1:
                st.markdown("---")
    
    else:
        st.markdown("**Showing all runs with best configurations highlighted**")
        for algo in algos:
            st.markdown(f"#### {algo} Runs")
            algo_runs = [r for r in env_runs if r['algorithm'] == algo]
            
            for idx, run in enumerate(algo_runs):
                if run == best_per_algo.get(algo):
                    st.success(f"⭐ **BEST CONFIGURATION** - Run #{run.get('run_id', idx+1)}")
                render_run_card(run, idx)
                if idx < len(algo_runs) - 1:
                    st.markdown("---")
            st.markdown("---")


def render_algorithm_analysis_view():
    """Analyze how parameters affect algorithm performance"""
    
    st.markdown("### Algorithm Deep Dive")
    st.markdown("Analyze how hyperparameters affect performance")
    
    runs = st.session_state.training_runs
    algos = sorted({r['algorithm'] for r in runs})
    
    algo = st.selectbox("Select Algorithm", algos)
    algo_runs = [r for r in runs if r['algorithm'] == algo]
    
    st.markdown(f"**Found {len(algo_runs)} runs for {algo}**")
    
    envs = sorted({r['environment'] for r in algo_runs})
    env = st.selectbox("Filter by Environment", ["All Environments"] + envs)
    
    if env != "All Environments":
        algo_runs = [r for r in algo_runs if r['environment'] == env]
    
    st.markdown("---")
    
    mode = st.radio(
        "Display Mode:",
        ["Show All Parameter Variations", "Show Best Configuration Only"],
        horizontal=True
    )
    
    st.markdown("### Hyperparameter Impact Analysis")
    
    params = sorted({k for r in algo_runs for k in r['params'].keys()})
    
    if not params:
        st.warning("No hyperparameter data available")
        return
    
    param = st.selectbox("Select Hyperparameter", params)
    
    data = [(r['params'][param], r['avg_reward'], r['environment'], 
             datetime.fromtimestamp(r['timestamp']).strftime('%Y-%m-%d %H:%M'),
             r.get('run_id', 'N/A'))
            for r in algo_runs if param in r['params'] and r.get('avg_reward')]
    
    if data:
        vals, rewards, env_labels, times, run_ids = zip(*data)
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        unique_envs = sorted(set(env_labels))
        
        best_idx = rewards.index(max(rewards))
        best_id = run_ids[best_idx]
        
        for env_idx, env_name in enumerate(unique_envs):
            env_data = [(v, r, t, rid) for v, r, e, t, rid in data if e == env_name]
            e_vals, e_rewards, e_times, e_ids = zip(*env_data)
            
            sizes = [20 if rid == best_id else 12 for rid in e_ids]
            symbols = ['star' if rid == best_id else 'circle' for rid in e_ids]
            
            fig.add_trace(go.Scatter(
                x=e_vals, y=e_rewards, mode='markers', name=env_name,
                marker=dict(size=sizes, symbol=symbols, 
                           color=colors[env_idx % len(colors)],
                           line=dict(width=2, color='white')),
                text=[f"{t} (Run {rid})" for t, rid in zip(e_times, e_ids)]
            ))
        
        fig.update_layout(
            title=f"Impact of {param} on Performance (⭐ = Best)",
            xaxis_title=param, yaxis_title="Average Reward",
            height=500, template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("#### Statistical Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Min Value", f"{min(vals):.4f}" if isinstance(min(vals), float) else str(min(vals)))
        with col2:
            st.metric("Max Value", f"{max(vals):.4f}" if isinstance(max(vals), float) else str(max(vals)))
        with col3:
            st.metric("Best Reward", f"{max(rewards):.2f}")
        with col4:
            best_val = vals[best_idx]
            st.metric("Best Config", f"{best_val:.4f}" if isinstance(best_val, float) else str(best_val))
        
        st.markdown("---")
        st.success(f"**⭐ Best Configuration: Run {best_id}**")
        best_run = next(r for r in algo_runs if r.get('run_id') == best_id)
        
        st.markdown("**Best Hyperparameters:**")
        cols = st.columns(min(len(best_run['params']), 4))
        for i, (k, v) in enumerate(best_run['params'].items()):
            with cols[i % 4]:
                st.write(f"**{k}:** {v:.4f}" if isinstance(v, float) else f"**{k}:** {v}")
    
    else:
        st.warning(f"No data available for parameter: {param}")
    
    st.markdown("---")
    
    if mode == "Show Best Configuration Only":
        st.markdown("### Best Configuration Details")
        reward_runs = [r for r in algo_runs if r.get('avg_reward')]
        if reward_runs:
            best = max(reward_runs, key=lambda x: x['avg_reward'])
            st.markdown(f"**Showing the best performing configuration (Run #{best.get('run_id', 'N/A')})**")
            render_run_card(best, 0)
        else:
            st.info("No runs with reward data available")
    
    else:
        st.markdown("### All Runs for This Algorithm")
        st.markdown(f"**Showing all {len(algo_runs)} runs**")
        for idx, run in enumerate(algo_runs):
            render_run_card(run, idx)
            if idx < len(algo_runs) - 1:
                st.markdown("---")


def render_comparison_view():
    """Compare multiple training runs side by side"""
    
    st.markdown("### Compare Training Runs")
    st.markdown("Select multiple runs to compare their performance and configurations")
    
    runs = st.session_state.training_runs
    
    st.markdown("#### Select Runs to Compare")
    
    options = []
    for idx, r in enumerate(runs):
        ts = datetime.fromtimestamp(r['timestamp']).strftime('%Y-%m-%d %H:%M')
        options.append({
            'ID': r.get('run_id', idx+1),
            'Algorithm': r['algorithm'],
            'Environment': r['environment'],
            'Avg Reward': r.get('avg_reward', 'N/A'),
            'Time': ts,
            'Index': idx
        })
    
    df = pd.DataFrame(options)
    st.dataframe(df[['ID', 'Algorithm', 'Environment', 'Avg Reward', 'Time']], use_container_width=True)
    
    ids = st.multiselect("Select Run IDs to Compare", df['ID'].tolist())
    
    if not ids:
        st.info("Select at least one run to compare")
        return
    
    if len(ids) > 6:
        st.warning("Comparing more than 6 runs may be difficult to visualize.")
    
    selected = [runs[df[df['ID'] == i]['Index'].values[0]] for i in ids]
    
    st.markdown("---")
    st.markdown("### Comparison Results")
    st.markdown("#### Performance Metrics")
    
    metrics = [{
        'Run ID': r.get('run_id', 'N/A'),
        'Algorithm': r['algorithm'],
        'Environment': r['environment'],
        'Avg Reward': r.get('avg_reward', 'N/A'),
        'Training Time (s)': r.get('training_time', 'N/A'),
        'Episodes/Iterations': r.get('episodes', r.get('iterations', 'N/A'))
    } for r in selected]
    
    st.dataframe(pd.DataFrame(metrics), use_container_width=True)
    
    if all('history' in r and 'episode_rewards' in r['history'] for r in selected):
        st.markdown("#### Reward Progression Comparison")
        
        fig = go.Figure()
        colors = px.colors.qualitative.Set1
        
        for idx, r in enumerate(selected):
            rewards = r['history']['episode_rewards']
            episodes = list(range(1, len(rewards) + 1))
            
            window = min(50, len(rewards) // 10) or 1
            avg = pd.Series(rewards).rolling(window, min_periods=1).mean()
            
            label = f"Run {r.get('run_id', idx+1)}: {r['algorithm']} ({r['environment']})"
            
            fig.add_trace(go.Scatter(
                x=episodes, y=rewards, mode='lines', name=label,
                line=dict(color=colors[idx % len(colors)], width=1),
                opacity=0.3, showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=episodes, y=avg, mode='lines', name=label,
                line=dict(color=colors[idx % len(colors)], width=3)
            ))
        
        fig.update_layout(
            title="Reward Progression Comparison",
            xaxis_title="Episode", yaxis_title="Reward",
            hovermode='x unified', height=600, template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Hyperparameter Comparison")
    
    all_params = {k for r in selected for k in r['params'].keys()}
    
    param_comp = []
    for p in sorted(all_params):
        row = {'Parameter': p}
        for r in selected:
            rid = r.get('run_id', 'N/A')
            val = r['params'].get(p, 'N/A')
            row[f"Run {rid}"] = f"{val:.4f}" if isinstance(val, float) else str(val)
        param_comp.append(row)
    
    st.dataframe(pd.DataFrame(param_comp), use_container_width=True)
    
    st.markdown("#### Video Comparison")
    
    tabs = st.tabs([f"Run {r.get('run_id', idx+1)}" for idx, r in enumerate(selected)])
    
    for idx, (tab, r) in enumerate(zip(tabs, selected)):
        with tab:
            st.markdown(f"**{r['algorithm']} on {r['environment']}**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Training Progress**")
                if r.get('training_video_path') and os.path.exists(r['training_video_path']):
                    st.video(r['training_video_path'], format='video/mp4')
                else:
                    st.info("Video not available")
            
            with col2:
                st.markdown("**Final Policy**")
                if r.get('inference_video_path') and os.path.exists(r['inference_video_path']):
                    st.video(r['inference_video_path'], format='video/mp4')
                else:
                    st.info("Video not available")
