"""
File: rl_env.py
Description: The Gym-like environment that adapts the reward function
             based on the specific Expert Agent (Time, Energy, or Security).
"""
import numpy as np
import config
from environment import calculate_cost

class OffloadingEnv:
    def __init__(self, network, workflow, objective="time"):
        """
        objective: "time", "energy", or "security"
        """
        self.network = network
        self.workflow = workflow
        self.objective = objective
        
        self.N = workflow.N
        self.action_space_n = network.total_locations
        
        # State tracking
        self.current_step = 1
        self.current_solution_p = [0] * self.N
        
        # Initial Metrics
        self.current_time = 0
        self.current_energy = 0
        self.current_risk = 0
        self._update_metrics()

    def _get_security_risk(self, solution_p):
        """
        Calculates a 'Security Risk Score'.
        - IoT (Loc 0): Risk 0.0 (Local is safe)
        - Edge: Risk 0.2 (Private network, relatively safe)
        - Cloud: Risk 1.0 (Public internet, highest exposure)
        """
        total_risk = 0
        
        # Map solution p (tasks 1..N) to locations
        # solution_p is a list of location IDs
        for loc in solution_p:
            if loc == 0:
                risk = 0.0
            elif loc <= self.network.num_edge:
                risk = 0.2
            else:
                risk = 1.0
            total_risk += risk
            
        return total_risk

    def _update_metrics(self):
        """Calculates T, E, and Risk for the current solution."""
        _, T, E = calculate_cost(self.workflow, self.current_solution_p, self.network)
        Risk = self._get_security_risk(self.current_solution_p)
        
        self.current_time = T
        self.current_energy = E
        self.current_risk = Risk

    def _action_transformation(self, p_old, h, action):
        """Applies action to tasks h through N (The Paper's Logic)"""
        p_new = list(p_old)
        # Apply action to current task 'h' and all future tasks
        # h is 1-indexed, list is 0-indexed
        for i in range(h - 1, self.N):
            p_new[i] = action
        return p_new

    def reset(self):
        self.current_step = 1
        self.current_solution_p = [0] * self.N # Reset to IoT-only
        self._update_metrics()
        return self._get_state()

    def _get_state(self):
        # State vector: [step, p_0, p_1, ... p_N-1]
        state = np.zeros(self.N + 1, dtype=np.float32)
        state[0] = self.current_step
        state[1:] = self.current_solution_p
        return state

    def step(self, action):
        if self.current_step > self.N:
            raise Exception("Episode done")

        # 1. Capture Old Metrics
        old_time = self.current_time
        old_energy = self.current_energy
        old_risk = self.current_risk

        # 2. Execute Action (Transition)
        p_new = self._action_transformation(self.current_solution_p, self.current_step, action)
        self.current_solution_p = p_new
        
        # 3. Update Metrics (Get New T, E, Risk)
        self._update_metrics()
        
        # 4. CALCULATE REWARD (The Expert Logic)
        # Reward is always (Old_Cost - New_Cost). 
        # If cost goes down, reward is positive.
        
        if self.objective == "time":
            reward = old_time - self.current_time
            
        elif self.objective == "energy":
            reward = old_energy - self.current_energy
            
        elif self.objective == "security":
            reward = old_risk - self.current_risk
            
        else: # Balanced/Default
            # Normalize them to combine? For simplicity, we use weighted sum
            # But usually experts optimize ONE thing.
            reward = (old_time - self.current_time) + (old_energy - self.current_energy)

        # 5. Advance Step
        self.current_step += 1
        done = self.current_step > self.N
        
        return self._get_state(), reward, done