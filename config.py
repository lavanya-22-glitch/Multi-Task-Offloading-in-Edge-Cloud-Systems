"""
Global Configuration for the BTP Project.
"""
import torch

# --- System Settings ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Environment Parameters (Based on Table II) ---
NUM_EDGE_SERVERS = 2
NUM_CLOUD_SERVERS = 1
NUM_TASKS = 4

# --- RL Hyperparameters ---
BATCH_SIZE = 32
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
MEMORY_CAPACITY = 5000
LEARNING_RATE = 0.001

# --- LLM Manager Settings ---
LLM_UPDATE_INTERVAL = 50  # Run LLM update every 50 episodes (Simulated "Slow Mind")
DEFAULT_WEIGHTS = {"energy": 0.5, "time": 0.5}

# --- Cost Coefficients ---
COST_TIME = 0.2
COST_ENERGY = 1.34