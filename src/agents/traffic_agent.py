import numpy as np

class TrafficAgent:
    """
    Represents a Smart Traffic Controller (STC) agent responsible for managing traffic flow.
    """
    def __init__(self, agent_id, intersection_id, initial_signal_phase=0):
        self.agent_id = agent_id
        self.intersection_id = intersection_id
        self.current_signal_phase = initial_signal_phase # Represents the active traffic light phase
        self.history = [] # To store past states and actions for learning

    def get_observation(self, local_traffic_params, local_pm25_from_sensors):
        """
        Returns the agent's local observation based on traffic sensors and nearby PM2.5 data.
        local_traffic_params could include: {'queue_length': X, 'avg_speed': Y, 'vehicle_count': Z}
        """
        # In a real system, this would come from physical sensors.
        # For simulation, it's provided by the environment.
        return {
            'queue_length': local_traffic_params.get('queue_length', 0),
            'avg_speed': local_traffic_params.get('avg_speed', 0),
            'vehicle_count': local_traffic_params.get('vehicle_count', 0),
            'local_pm25': local_pm25_from_sensors,
            'current_signal_phase': self.current_signal_phase
        }

    def select_action(self, observation, policy_network=None):
        """
        Selects an action based on the current observation and a policy network.
        If no policy_network is provided, it can implement a simple rule-based or random policy.

        Actions:
        - Adjust traffic light timings (e.g., extend green, shorten red)
        - Change signal sequence (e.g., skip a phase)
        - Suggest alternative routes (conceptual, might influence reward)
        """
        if policy_network:
            # In a full MARL implementation, this would involve feeding the observation
            # through the agent's local actor network to get an action distribution.
            # For now, we'll simulate a simple decision.
            action_probs = policy_network.predict(observation)
            # Sample action from action_probs
            # For simplicity, let's assume action is a tuple: (delta_green_time, next_phase_override)
            delta_green_time = np.random.uniform(-5, 5) # Example: adjust green time by -5 to +5 seconds
            next_phase_override = np.random.choice([None, 0, 1, 2, 3]) # Example: potentially override next phase
        else:
            # Simple rule-based action for demonstration
            delta_green_time = 0
            next_phase_override = None

            if observation['queue_length'] > 50 and observation['local_pm25'] > 25: # High congestion and pollution
                delta_green_time = 10 # Extend green time to clear traffic
            elif observation['queue_length'] < 10 and observation['local_pm25'] < 15: # Low congestion and pollution
                delta_green_time = -5 # Shorten green time to optimize flow elsewhere

            # Example: if PM2.5 is very high, try to clear traffic quickly
            if observation['local_pm25'] > 40:
                next_phase_override = (self.current_signal_phase + 1) % 4 # Move to next phase faster

        # For simplicity, we'll just return the chosen actions.
        return {'delta_green_time': delta_green_time, 'next_phase_override': next_phase_override}

    def update(self, observation, action, reward, next_observation):
        """
        Updates the agent's internal state or learning model based on feedback.
        In a MARL setting, this would involve storing experiences and potentially
        updating the local actor network.
        """
        self.history.append((observation, action, reward, next_observation))
        # In a real MARL agent, the actor network would be updated here
        # based on the reward and next_observation, guided by the critic.
        pass

    def __repr__(self):
        return f"TrafficAgent(id={self.agent_id}, intersection={self.intersection_id}, phase={self.current_signal_phase})"
