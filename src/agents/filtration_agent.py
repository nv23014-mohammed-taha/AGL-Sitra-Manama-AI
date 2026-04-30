import numpy as np

class FiltrationAgent:
    """
    Represents a Bio-Digital Curtain (BDC) agent responsible for local air filtration.
    """
    def __init__(self, agent_id, position, initial_fan_speed=0.5, initial_bio_activity=0.7):
        self.agent_id = agent_id
        self.position = np.array(position) # (x, y, z) coordinates
        self.fan_speed = initial_fan_speed # Normalized between 0 and 1
        self.bio_activity = initial_bio_activity # Normalized between 0 and 1
        self.history = [] # To store past states and actions for learning

    def get_observation(self, local_pm25, local_wind_vector, temperature, humidity):
        """
        Returns the agent's local observation based on environmental sensors.
        """
        # In a real system, this would come from physical sensors.
        # For simulation, it's provided by the environment.
        return {
            'local_pm25': local_pm25,
            'local_wind_vector': local_wind_vector, # (wind_x, wind_y, wind_z)
            'temperature': temperature,
            'humidity': humidity,
            'fan_speed': self.fan_speed,
            'bio_activity': self.bio_activity
        }

    def select_action(self, observation, policy_network=None):
        """
        Selects an action based on the current observation and a policy network.
        If no policy_network is provided (e.g., during initial testing), it can
        implement a simple rule-based or random policy.

        Actions:
        - Adjust fan speed (e.g., increase, decrease, maintain)
        - Toggle mechanical filter (on/off)
        - Optimize nutrient supply (affects bio_activity over time)
        """
        if policy_network:
            # In a full MARL implementation, this would involve feeding the observation
            # through the agent's local actor network to get an action distribution.
            # For now, we'll simulate a simple decision.
            action_probs = policy_network.predict(observation)
            # Sample action from action_probs
            # For simplicity, let's assume action is a tuple: (delta_fan_speed, filter_on, delta_nutrient)
            delta_fan_speed = np.random.uniform(-0.1, 0.1) # Example: small random adjustment
            filter_on = np.random.choice([0, 1]) # Example: random on/off
            delta_nutrient = np.random.uniform(-0.05, 0.05) # Example: small random adjustment
        else:
            # Simple rule-based action for demonstration
            delta_fan_speed = 0
            filter_on = 0
            delta_nutrient = 0

            if observation['local_pm25'] > 30: # If PM2.5 is high
                delta_fan_speed = 0.2 # Increase fan speed
                filter_on = 1 # Turn on mechanical filter
            elif observation['local_pm25'] < 10: # If PM2.5 is low
                delta_fan_speed = -0.1 # Decrease fan speed
                filter_on = 0 # Turn off mechanical filter

            # Adjust bio-activity based on light/temp (simplified)
            if observation['temperature'] > 25 and observation['light_intensity'] > 0.5: # Assuming light_intensity is part of observation
                delta_nutrient = 0.02
            else:
                delta_nutrient = -0.01

        # Apply action and clip values to stay within bounds
        self.fan_speed = np.clip(self.fan_speed + delta_fan_speed, 0, 1)
        # For filter_on, it's a discrete state, so we just set it.
        # bio_activity changes over time based on nutrient and environmental factors, not directly by action
        # For now, we'll just return the chosen actions.
        return {'delta_fan_speed': delta_fan_speed, 'filter_on': filter_on, 'delta_nutrient': delta_nutrient}

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
        return f"FiltrationAgent(id={self.agent_id}, pos={self.position}, fan_speed={self.fan_speed:.2f}, bio_activity={self.bio_activity:.2f})"
