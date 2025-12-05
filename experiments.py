"""
File: experiments.py
Description: Runs the simulation and generates graphs for the Research Paper.
             1. Convergence Plot (Did the experts learn?)
             2. Adaptation Plot (Did the Manager switch strategies?)
"""
import matplotlib.pyplot as plt
import numpy as np
import config
from environment import Network, Workflow
from rl_env import OffloadingEnv
from experts import ExpertAgent
from gating import LLM_Manager
from main import get_random_context # Re-using helper

def run_experiment():
    print("--- Running Experiment for Graphs ---")
    
    # Setup System (Same as main.py)
    network = Network(randomize=True)
    task_sizes = [1, 10e6, 15e6, 8e6, 12e6, 1]
    dependencies = [(0, 1, 1e5), (1, 2, 5e5), (1, 3, 3e5), (2, 4, 4e5), (3, 4, 6e5), (4, 5, 5e5)]
    workflow = Workflow(config.NUM_TASKS, task_sizes, dependencies)
    env = OffloadingEnv(network, workflow, objective="balanced")
    
    state_size = env.N + 1
    action_size = env.action_space_n
    
    # Agents & Manager
    experts = {
        "time": ExpertAgent(state_size, action_size),
        "energy": ExpertAgent(state_size, action_size),
        "security": ExpertAgent(state_size, action_size)
    }
    manager = LLM_Manager()
    
    # --- DATA COLLECTION ---
    history_rewards = []
    history_weights_time = []
    history_weights_energy = []
    history_weights_security = []
    history_context_change = [] # Mark episodes where context changed

    total_episodes = 150 # Enough to see changes
    
    for episode in range(1, total_episodes + 1):
        
        # Manager Update Logic
        if episode % 30 == 0: # Force update every 30 eps
            context = get_random_context()
            manager.update_strategy(context)
            history_context_change.append(episode)
            
        weights = manager.get_weights()
        
        # Log Weights
        history_weights_time.append(weights.get("time", 0))
        history_weights_energy.append(weights.get("energy", 0))
        history_weights_security.append(weights.get("security", 0))

        # Run Episode
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Voting Logic
            vote_scores = np.zeros(action_size)
            for name, agent in experts.items():
                action = agent.choose_action(state)
                vote_scores[action] += weights.get(name, 0)
            
            final_action = np.argmax(vote_scores)
            
            # Metrics for Rewards
            prev_time, prev_energy, prev_risk = env.current_time, env.current_energy, env.current_risk
            next_state, _, done = env.step(final_action)
            
            # Calculate Expert Rewards
            rewards = {
                "time": prev_time - env.current_time,
                "energy": prev_energy - env.current_energy,
                "security": prev_risk - env.current_risk
            }
            
            # Train
            for name, agent in experts.items():
                agent.memory.push(state, final_action, rewards[name], next_state, done)
                agent.learn()
                
            state = next_state
            episode_reward += sum(rewards.values()) # Sum of all expert rewards
            
        history_rewards.append(episode_reward)
        if episode % 10 == 0: print(f"Exp Episode {episode} done.")

    return history_rewards, history_weights_time, history_weights_energy, history_weights_security, history_context_change

def plot_results(rewards, w_time, w_energy, w_sec, changes):
    """Generates and saves the Research Plots."""
    
    epochs = range(1, len(rewards) + 1)
    
    # --- PLOT 1: Learning Convergence ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, rewards, label='System Reward', color='blue', alpha=0.6)
    # Smooth line
    smooth_rewards = np.convolve(rewards, np.ones(10)/10, mode='valid')
    plt.plot(range(10, len(rewards)+1), smooth_rewards, label='Moving Avg (10)', color='red', linewidth=2)
    
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward (Improvement)')
    plt.title('Convergence of Hierarchical MoE System')
    plt.legend()
    plt.grid(True)
    plt.savefig('result_convergence.png')
    print("Saved 'result_convergence.png'")
    
    # --- PLOT 2: Strategy Adaptation (The "Agentic" Proof) ---
    plt.figure(figsize=(10, 5))
    plt.stackplot(epochs, w_time, w_energy, w_sec, labels=['Time Weight', 'Energy Weight', 'Security Weight'], alpha=0.8)
    
    # Draw lines where context changed
    for c in changes:
        plt.axvline(x=c, color='black', linestyle='--', alpha=0.5)
        
    plt.xlabel('Episodes')
    plt.ylabel('Strategy Weight Allocation')
    plt.title('Dynamic Strategy Adaptation by Manager (LLM)')
    plt.legend(loc='upper left')
    plt.savefig('result_adaptation.png')
    print("Saved 'result_adaptation.png'")
    
    plt.show()

if __name__ == "__main__":
    data = run_experiment()
    plot_results(*data)