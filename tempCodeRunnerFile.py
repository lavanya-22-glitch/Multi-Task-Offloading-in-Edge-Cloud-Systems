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
