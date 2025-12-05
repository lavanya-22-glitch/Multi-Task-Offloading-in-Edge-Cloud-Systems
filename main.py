"""
File: main.py
Description: The Main Training Loop.
             Implements the 'Slow Mind / Fast Mind' architecture
             and the Weighted Voting mechanism.
"""
import numpy as np
import random
import config
from environment import Network, Workflow, calculate_cost
from rl_env import OffloadingEnv
from experts import ExpertAgent
from gating import LLM_Manager

def get_random_context():
    """Generates a random context for the LLM Manager."""
    return {
        "battery": random.randint(10, 100),
        "activity": random.choice(["idle", "gaming", "banking", "video_call"]),
        "network_load": random.choice(["low", "high"])
    }

def main():
    print("--- Initialize System ---")
    
    # 1. Create the Physical World
    network = Network(randomize=True)
    # Define a simple DAG (Fig 3a style)
    task_sizes = [1, 10e6, 15e6, 8e6, 12e6, 1]
    dependencies = [
        (0, 1, 1e5), (1, 2, 5e5), (1, 3, 3e5), 
        (2, 4, 4e5), (3, 4, 6e5), (4, 5, 5e5)
    ]
    workflow = Workflow(config.NUM_TASKS, task_sizes, dependencies)
    
    # 2. Create the Game Environment (Neutral Objective)
    env = OffloadingEnv(network, workflow, objective="balanced")
    state_size = env.N + 1
    action_size = env.action_space_n

    # 3. Hire the Experts (The Workers)
    print("--- Hiring Experts ---")
    agent_time = ExpertAgent(state_size, action_size, name="Time_Expert")
    agent_energy = ExpertAgent(state_size, action_size, name="Energy_Expert")
    agent_security = ExpertAgent(state_size, action_size, name="Security_Expert")
    
    experts = {
        "time": agent_time,
        "energy": agent_energy,
        "security": agent_security
    }

    # 4. Hire the Manager (The Brain)
    print("--- Initializing LLM Manager ---")
    manager = LLM_Manager()

    # --- MAIN LOOP ---
    print(f"\n--- Starting Training ({config.BATCH_SIZE} episodes) ---")
    
    total_episodes = 200 # Short run for demo
    
    for episode in range(1, total_episodes + 1):
        
        # --- A. THE SLOW LOOP (Strategy Update) ---
        # Runs periodically to simulate context changes
        if episode % config.LLM_UPDATE_INTERVAL == 0:
            context = get_random_context()
            manager.update_strategy(context)
            
        # Get current strategy weights (e.g., {'time': 0.8, ...})
        weights = manager.get_weights()

        # --- B. THE FAST LOOP (Execution) ---
        state = env.reset()
        done = False
        total_system_reward = 0
        
        while not done:
            # 1. ENSEMBLE INFERENCE (All Experts Vote)
            # Each expert suggests an action based on their own goal
            votes = {}
            for name, agent in experts.items():
                votes[name] = agent.choose_action(state)
            
            # 2. WEIGHTED VOTING (The Gating Logic)
            # We tally votes based on Manager's weights
            vote_scores = np.zeros(action_size)
            
            for name, action in votes.items():
                w = weights.get(name, 0)
                vote_scores[action] += w
                
            # The winner is the action with the highest weighted score
            final_action = np.argmax(vote_scores)
            
            # 3. EXECUTE (Physical Reality)
            # We assume the experts acted, but we only execute the Winner.
            
            # Capture state before move to calc rewards manually
            prev_time = env.current_time
            prev_energy = env.current_energy
            prev_risk = env.current_risk
            
            # Apply to environment
            next_state, _, done = env.step(final_action)
            
            # 4. DUAL-LOOP FEEDBACK (Calculate Rewards)
            # We calculate what the reward WOULD have been for each expert
            # This trains them even if their action wasn't chosen (Off-Policy)
            
            r_time = prev_time - env.current_time
            r_energy = prev_energy - env.current_energy
            r_security = prev_risk - env.current_risk
            
            rewards = {"time": r_time, "energy": r_energy, "security": r_security}
            
            # 5. TRAIN EXPERTS
            for name, agent in experts.items():
                # Each agent learns from the SAME transition, 
                # but receives their SPECIFIC reward.
                r = rewards[name]
                agent.memory.push(state, final_action, r, next_state, done)
                agent.learn()
                
            state = next_state
            
            # (Optional) Track total system score
            total_system_reward += sum(rewards.values())

        # Progress Report
        if episode % 10 == 0:
            print(f"Ep {episode}: Strategy={weights} | Winner Action={final_action}")

    print("\n--- Training Complete ---")
    print("The system is now ready for deployment.")

    # -------------------------------------------------------------
    # --- 5. FINAL INFERENCE (Testing the Policy) ---
    # This code is now INSIDE main(), so it can see 'manager'
    # -------------------------------------------------------------
    print("\n--- 5. FINAL INFERENCE (Testing the Policy) ---")
    print("Let's see the actual decisions for a sample Workflow...")

    # Define test data (re-using the structure from training)
    test_task_sizes = [1, 10e6, 15e6, 8e6, 12e6, 1]
    test_dependencies = [
        (0, 1, 1e5), (1, 2, 5e5), (1, 3, 3e5), 
        (2, 4, 4e5), (3, 4, 6e5), (4, 5, 5e5)
    ]

    # 1. Create a Test Workflow
    test_network = Network(randomize=False) # Use static/stable network for testing
    test_workflow = Workflow(config.NUM_TASKS, test_task_sizes, test_dependencies)
    
    # 2. Reset Environment
    test_env = OffloadingEnv(test_network, test_workflow, objective="balanced")
    state = test_env.reset()
    done = False
    
    # 3. Get Strategy from Manager
    current_weights = manager.get_weights() 
    print(f"Current Strategy Weights: {current_weights}")

    print("\n--- Task-by-Task Decisions ---")
    
    # 4. Run the "Fast Mind" Loop to generate the Policy
    task_counter = 1
    while not done:
        # A. Experts Vote
        votes = {}
        print(f"Task {task_counter}:")
        for name, agent in experts.items():
            # Note: We set evaluate=True to stop random exploration
            action = agent.choose_action(state, evaluate=True) 
            votes[name] = action
            print(f"  - {name} Agent votes for: Location {action}")

        # B. Weighted Voting
        vote_scores = np.zeros(action_size)
        for name, action in votes.items():
            vote_scores[action] += current_weights.get(name, 0)
        
        final_action = np.argmax(vote_scores)
        print(f"  >> FINAL DECISION: Location {final_action}")

        # C. Execute
        state, _, done = test_env.step(final_action)
        task_counter += 1

    # 5. Result
    final_policy = test_env.current_solution_p
    print(f"\n[End Result] Optimized Offloading Policy p: {final_policy}")
    
    # Decode for readability
    location_names = ["IoT Device"] + \
                     [f"Edge {i+1}" for i in range(config.NUM_EDGE_SERVERS)] + \
                     [f"Cloud {i+1}" for i in range(config.NUM_CLOUD_SERVERS)]
    
    print("\n[Human Readable Plan]")
    for i, loc_id in enumerate(final_policy):
        print(f"  Task {i+1} --> {location_names[loc_id]}")

if __name__ == "__main__":
    main()