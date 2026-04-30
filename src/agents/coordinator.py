import numpy as np

class Coordinator:
    """
    The Central Coordination Module (CCM) for the AGL framework.
    During training, it provides a centralized view for the critic.
    During decentralized execution, it can monitor overall performance and facilitate communication.
    """
    def __init__(self, num_filtration_agents, num_traffic_agents):
        self.num_filtration_agents = num_filtration_agents
        self.num_traffic_agents = num_traffic_agents
        self.global_history = [] # To store global states, joint actions, and rewards

    def get_global_observation_for_critic(self, cfd_simulator_global_state):
        """
        Aggregates relevant information from the CFD simulator's global state
        into a format suitable for the centralized critic network.
        This would involve flattening and potentially normalizing various fields.
        """
        pm25_flat = cfd_simulator_global_state["pm25_field"].flatten()
        wind_flat = cfd_simulator_global_state["wind_field"].flatten()
        temp_flat = cfd_simulator_global_state["temperature_field"].flatten()
        humidity_flat = cfd_simulator_global_state["humidity_field"].flatten()

        # Traffic data needs careful flattening as it's a dict of dicts
        traffic_flat = []
        for intersection_id in sorted(cfd_simulator_global_state["traffic_data"].keys()):
            traffic_data_item = cfd_simulator_global_state["traffic_data"][intersection_id]
            traffic_flat.append(traffic_data_item.get("queue_length", 0))
            traffic_flat.append(traffic_data_item.get("avg_speed", 0))
            traffic_flat.append(traffic_data_item.get("vehicle_count", 0))

        # Combine all flattened arrays into a single global state vector
        global_obs_vector = np.concatenate([
            pm25_flat,
            wind_flat,
            temp_flat,
            humidity_flat,
            np.array(traffic_flat)
        ])
        return global_obs_vector

    def record_step(self, global_state, joint_actions, reward, next_global_state):
        """
        Records the global state, joint actions, and reward for a given step.
        Useful for centralized training and analysis.
        """
        self.global_history.append({
            "global_state": global_state,
            "joint_actions": joint_actions,
            "reward": reward,
            "next_global_state": next_global_state
        })

    def get_global_metrics(self, cfd_simulator):
        """
        Calculates and returns key global metrics for monitoring.
        """
        global_state = cfd_simulator.get_global_state()
        avg_pm25 = np.mean(global_state["pm25_field"])
        pwe = cfd_simulator.calculate_population_weighted_exposure(global_state["pm25_field"])
        avg_traffic_latency = cfd_simulator.calculate_average_traffic_latency(global_state["traffic_data"])
        # Note: Energy consumption requires agent actions, not just global state

        return {
            "average_pm25": avg_pm25,
            "population_weighted_exposure": pwe,
            "average_traffic_latency": avg_traffic_latency
        }

