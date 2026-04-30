import numpy as np

class MultiObjectiveReward:
    """
    Defines the multi-objective reward function for the AGL framework.
    This function balances public health (PM2.5 exposure), economic efficiency
    (traffic latency), and operational costs (energy consumption).
    """
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
        """
        Initializes the reward function with weighting factors.
        :param alpha: Weight for Population-Weighted Exposure (PWE)
        :param beta: Weight for Traffic Latency Deviation (TLD)
        :param gamma: Weight for Energy Consumption (EC)
        """
        if not (0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1):
            raise ValueError("Weights (alpha, beta, gamma) must be between 0 and 1.")
        if not np.isclose(alpha + beta + gamma, 1.0):
            print("Warning: Weights do not sum to 1.0. They will be normalized.")
            total_weight = alpha + beta + gamma
            self.alpha = alpha / total_weight
            self.beta = beta / total_weight
            self.gamma = gamma / total_weight
        else:
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma

        print(f"Reward function initialized with weights: alpha={self.alpha:.2f}, beta={self.beta:.2f}, gamma={self.gamma:.2f}")

    def calculate_reward(self, current_global_state, next_global_state, filtration_actions, traffic_actions, cfd_simulator):
        """
        Calculates the overall reward for a given step.

        The reward function is defined as:
        R = - (alpha * PWE + beta * TLD + gamma * EC)

        Where:
        - PWE: Population-Weighted Exposure to PM2.5
        - TLD: Traffic Latency Deviation (penalty for increased traffic latency)
        - EC: Energy Consumption of BDCs

        :param current_global_state: The state of the environment before actions were taken.
        :param next_global_state: The state of the environment after actions were taken.
        :param filtration_actions: Dictionary of actions taken by filtration agents.
        :param traffic_actions: Dictionary of actions taken by traffic agents.
        :param cfd_simulator: An instance of the CFDSimulator to calculate metrics.
        :return: The scalar reward value.
        """

        # Calculate Population-Weighted Exposure (PWE) from the next state
        # Lower PWE is better, so we want to minimize this.
        pwe = cfd_simulator.calculate_population_weighted_exposure(next_global_state["pm25_field"])

        # Calculate Traffic Latency Deviation (TLD)
        # We need to compare current latency with next latency or a baseline.
        # For simplicity, let's just use the next state's latency as a cost.
        traffic_latency = cfd_simulator.calculate_average_traffic_latency(next_global_state["traffic_data"])
        # A simple deviation could be (traffic_latency - baseline_latency). For now, just use latency as cost.
        tld = traffic_latency # Assuming baseline is 0 or implicitly handled by scaling

        # Calculate Energy Consumption (EC)
        ec = cfd_simulator.calculate_energy_consumption(filtration_actions)

        # Normalize these values if they are on very different scales
        # For now, we'll assume they are somewhat scaled or the weights handle it.
        # In a real system, you'd want to normalize these costs to prevent one from dominating.
        # Example normalization (conceptual):
        # pwe_normalized = pwe / MAX_PWE
        # tld_normalized = tld / MAX_TLD
        # ec_normalized = ec / MAX_EC

        # Combine costs into a single objective to minimize
        total_cost = (self.alpha * pwe) + (self.beta * tld) + (self.gamma * ec)

        # Reward is the negative of the total cost (we want to maximize reward, so minimize cost)
        reward = -total_cost

        return reward

    def get_weights(self):
        return {"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma}
