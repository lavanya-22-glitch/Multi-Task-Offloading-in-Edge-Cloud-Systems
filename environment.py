"""
File: environment.py
Description: Defines the physical world (Servers, DAGs) and the physics (Cost Calculation).
"""
import numpy as np
import collections
import config  # Importing your settings

class Network:
    """
    Represents the Edge-Cloud Network.
    """
    def __init__(self, randomize=True):
        self.num_edge = config.NUM_EDGE_SERVERS
        self.num_cloud = config.NUM_CLOUD_SERVERS
        self.total_locations = 1 + self.num_edge + self.num_cloud
        
        # Initialize matrices with zeros
        self.V_R = np.zeros(self.total_locations) # Processing Speed
        self.V_E = np.zeros(self.total_locations) # Energy Consumption
        self.D_R = np.zeros((self.total_locations, self.total_locations)) # Transfer Speed
        self.D_E = np.zeros(self.total_locations) # Transfer Energy

        if randomize:
            self._randomize_environment()
        else:
            self._set_static_environment()

    # def _randomize_environment(self):
    #     # Randomized parameters for Meta-Learning
    #     # IoT Device (Loc 0)
    #     self.V_R[0] = np.random.uniform(0.010, 0.015)
        
    #     # Edge Servers (Loc 1 to E)
    #     self.V_R[1:1+self.num_edge] = np.random.uniform(0.005, 0.009)
        
    #     # Cloud Servers (Loc E+1 to End)
    #     self.V_R[1+self.num_edge:] = np.random.uniform(0.002, 0.005)
        
    #     # Energy Costs (Fixed roughly based on type)
    #     self.V_E[0] = 2.0
    #     self.V_E[1:1+self.num_edge] = 1.0
    #     self.V_E[1+self.num_edge:] = 0.5

    #     # Data Transfer Speeds (Simulated Latency)
    #     # 1. IoT <-> Edge
    #     iot_edge = np.random.uniform(2.0, 2.5)
    #     self.D_R[0, 1:1+self.num_edge] = iot_edge
    #     self.D_R[1:1+self.num_edge, 0] = iot_edge
        
    #     # 2. Edge <-> Cloud
    #     edge_cloud = np.random.uniform(1.0, 1.5)
    #     self.D_R[1:1+self.num_edge, 1+self.num_edge:] = edge_cloud
    #     self.D_R[1+self.num_edge:, 1:1+self.num_edge] = edge_cloud
        
    #     np.fill_diagonal(self.D_R, 0)
        
    #     # Data Energy
    #     self.D_E[0] = 0.1
    #     self.D_E[1:1+self.num_edge] = 0.05
    #     self.D_E[1+self.num_edge:] = 0.02


    # def _randomize_environment(self):
    #     # --- THE TWIST: Weak Hardware ---
    #     # 1. IoT Device is 100x slower than normal
    #     self.V_R[0] = 1.2  # Normal was 0.012
        
    #     # 2. IoT Device burns 10x energy per cycle
    #     self.V_E[0] = 20.0 # Normal was 2.0
        
    #     # 3. Network is excellent (Fiber optic speeds)
    #     self.D_R[:] = 0.1  # Super fast transfer everywhere
    #     self.D_E[:] = 0.01 # Cheap transfer
        
    #     # Keep Cloud/Edge speeds normal (fast)
    #     self.V_R[1:1+self.num_edge] = 0.007
    #     self.V_R[1+self.num_edge:] = 0.003


    def _randomize_environment(self):
        # Keep Task Speeds (V_R) normal
        self.V_R[0] = 0.012; self.V_R[1:1+self.num_edge] = 0.007; self.V_R[1+self.num_edge:] = 0.003
        
        # --- THE TWIST: Bad Internet ---
        # 1. Edge is nearby (Wi-Fi)
        self.D_R[0, 1:1+self.num_edge] = 1.0  # Fast
        self.D_E[1:1+self.num_edge] = 0.05    # Cheap
        
        # 2. Cloud is far away (Bad 5G)
        edge_cloud_latency = 50.0             # Huge Latency
        cloud_energy_cost = 5.0               # Huge Energy Drain
        
        self.D_R[1:1+self.num_edge, 1+self.num_edge:] = edge_cloud_latency
        self.D_R[1+self.num_edge:, 1:1+self.num_edge] = edge_cloud_latency
        self.D_E[1+self.num_edge:] = cloud_energy_cost
        
        np.fill_diagonal(self.D_R, 0)

    
    def _set_static_environment(self):
        # Original static values from the paper (for testing)
        self.V_R[0] = 0.012; self.V_R[1:1+self.num_edge] = 0.007; self.V_R[1+self.num_edge:] = 0.003
        self.V_E[0] = 2.0; self.V_E[1:1+self.num_edge] = 1.0; self.V_E[1+self.num_edge:] = 0.5
        # (Simplified static D_R setup...)
        self.D_R[:] = 1.2
        np.fill_diagonal(self.D_R, 0)

class Workflow:
    """
    Represents the Application DAG (Task Dependencies).
    """
    def __init__(self, num_tasks, task_sizes, dependencies):
        self.N = num_tasks
        self.num_nodes = num_tasks + 2 # +2 for Entry/Exit nodes
        
        self.task_sizes = np.array(task_sizes)
        self.adj = collections.defaultdict(list)
        self.rev_adj = collections.defaultdict(list)
        self.data_sizes = {}

        for (u, v, data_size) in dependencies:
            self.adj[u].append(v)
            self.rev_adj[v].append(u)
            self.data_sizes[(u, v)] = data_size

    def get_children(self, task_id): return self.adj[task_id]
    def get_parents(self, task_id): return self.rev_adj[task_id]
    def get_data_size(self, u, v): return self.data_sizes.get((u, v), 0)

def calculate_cost(workflow, solution_p, network):
    """
    The Physics Engine: Calculates Time (T) and Energy (E).
    Returns: (Total_Cost, Time, Energy)
    """
    N = workflow.N
    
    # Map solution to full location array (including entry/exit)
    locations = np.zeros(N + 2, dtype=int)
    locations[1:N+1] = solution_p 
    
    # 1. Calculate Energy (E)
    E_V = 0
    for i in range(1, N + 1):
        E_V += workflow.task_sizes[i] * network.V_E[locations[i]]
        
    E_D = 0
    for i in range(1, N + 1): 
        l_i = locations[i]
        D_E_li = network.D_E[l_i]
        
        data_io = 0
        for j in workflow.get_parents(i):
            data_io += workflow.get_data_size(j, i)
        for k in workflow.get_children(i):
            data_io += workflow.get_data_size(i, k)
            
        E_D += D_E_li * data_io
        
    E = network.V_E[0] * (E_V + E_D) # Scaling by base energy unit

    # 2. Calculate Time (T) via Critical Path
    # Topological Sort
    in_degree = collections.defaultdict(int)
    for u in range(workflow.num_nodes):
        for v in workflow.get_children(u):
            in_degree[v] += 1
            
    queue = collections.deque([0])
    topo_order = []
    while queue:
        u = queue.popleft()
        topo_order.append(u)
        for v in workflow.get_children(u):
            in_degree[v] -= 1
            if in_degree[v] == 0: queue.append(v)
            
    # Longest Path
    dist = collections.defaultdict(lambda: -float('inf'))
    dist[0] = 0
    
    Delta_max = 0
    for u in topo_order:
        if dist[u] == -float('inf'): continue
        
        # Exit node check
        if u == workflow.num_nodes - 1:
            dist[u] += workflow.task_sizes[u] * network.V_R[locations[u]]
            Delta_max = dist[u]
            break

        for v in workflow.get_children(u):
            # Task Execution Time
            v_i = workflow.task_sizes[u]
            l_i = locations[u]
            task_time = v_i * network.V_R[l_i]
            
            # Data Transfer Time
            d_ij = workflow.get_data_size(u, v)
            l_j = locations[v]
            transfer_time = d_ij * network.D_R[l_i, l_j]
            
            new_dist = dist[u] + task_time + transfer_time
            if new_dist > dist[v]:
                dist[v] = new_dist
                
    T = Delta_max
    
    # NOTE: We return Raw T and E. The "Weights" are applied by the Agents later.
    # Total generic cost for logging purposes only
    U = (config.COST_TIME * T) + (config.COST_ENERGY * E)
    
    return U, T, E