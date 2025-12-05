"""
File: gating.py
Description: The High-Level Manager (LLM Gating Network).
             It decides the 'Strategy Weights' based on context.
             In a real deployment, this calls the Gemini/Llama API.
"""
import random
import config
import json
import google.generativeai as genai

# --- API CONFIGURATION ---
# Put your API key here (Global Scope)
genai.configure(api_key="AIzaSyBx16XY3D06q8HR7Q4Sgwz5WKLGnJCx-aQ")

class LLM_Manager:
    def __init__(self):
        # Start with a balanced strategy
        self.current_weights = config.DEFAULT_WEIGHTS.copy()
        self.context_history = []
        
        # Initialize the Gemini Model once
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.api_available = True
        except Exception as e:
            print(f"[Manager] Warning: Gemini API not available ({e}). Using simulation.")
            self.api_available = False

    def get_weights(self):
        """
        Returns the current strategy weights.
        Called by the 'Fast Mind' (Main Loop) every millisecond.
        """
        return self.current_weights

    def update_strategy(self, current_context):
        """
        The 'Slow Mind' update. 
        Tries to call the REAL Gemini API first. 
        Falls back to simulation if API fails.
        """
        print(f"\n[Manager] Analyzing Context: {current_context}...")
        
        # Try using the Real API
        if self.api_available:
            try:
                # 1. Construct Semantic Prompt
                prompt = f"""
                You are an Edge Computing Resource Manager.
                Context: Battery is {current_context['battery']}%, Activity is {current_context['activity']}, Network Load is {current_context['network_load']}.
                
                Task: Return a VALID JSON object with weights for 'time', 'energy', and 'security'.
                They must sum to 1.0. Do not write markdown. Just the JSON.
                Example: {{"time": 0.5, "energy": 0.5, "security": 0.0}}
                """
                
                # 2. Call Gemini
                response = self.model.generate_content(prompt)
                
                # 3. Clean Response (Remove Markdown if present)
                text = response.text.replace("```json", "").replace("```", "").strip()
                
                # 4. Parse JSON
                new_weights = json.loads(text)
                
                self.current_weights = new_weights
                print(f"[Manager] New Strategy Set (from Gemini): {self.current_weights}\n")
                return # Success! Exit function.

            except Exception as e:
                print(f"[Manager] API Error: {e}. Falling back to simulation logic.")
        
        # --- FALLBACK: SIMULATED LOGIC ---
        # If API is disabled or fails, use this logic
        self.current_weights = self._simulated_llm_logic(current_context)
        print(f"[Manager] New Strategy Set (Simulated): {self.current_weights}\n")

    def _simulated_llm_logic(self, context):
        """
        A hard-coded 'Mock LLM' to demonstrate the logic.
        Used as a fallback if the API fails.
        """
        battery = context.get('battery', 100)
        activity = context.get('activity', 'idle')
        
        weights = {"time": 0.33, "energy": 0.33, "security": 0.33} # Default Balanced

        # Scenario A: Critical Battery -> Prioritize Energy
        if battery < 20:
            weights = {"time": 0.1, "energy": 0.8, "security": 0.1}
            print("  (Simulated Reason: Battery is critical! Saving power.)")

        # Scenario B: Gaming/Video -> Prioritize Time
        elif activity in ["gaming", "video_call"]:
            weights = {"time": 0.8, "energy": 0.1, "security": 0.1}
            print("  (Simulated Reason: User is Gaming. Latency is king.)")

        # Scenario C: Banking -> Prioritize Security
        elif activity == "banking":
            weights = {"time": 0.1, "energy": 0.1, "security": 0.8}
            print("  (Simulated Reason: Banking App detected. Maximum Security.)")
            
        return weights