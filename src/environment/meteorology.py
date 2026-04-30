import numpy as np
import random

class Meteorology:
    """
    Simulates meteorological conditions like wind, temperature, and humidity.
    This class provides realistic or configurable weather patterns for the CFD simulator.
    """
    def __init__(self, initial_wind_speed=5.0, initial_wind_direction=270, # 270 degrees is Westerly
                 initial_temperature=28.0, initial_humidity=0.6):
        self.current_wind_speed = initial_wind_speed
        self.current_wind_direction = initial_wind_direction # Degrees from North, clockwise
        self.current_temperature = initial_temperature
        self.current_humidity = initial_humidity
        self.time_of_day = 0 # Hours from midnight

    def _update_wind(self, time_step_hours):
        # Simulate daily and seasonal wind patterns, including Shamal winds
        # Shamal winds are typically north-westerly (around 270-315 degrees)
        # For simplicity, let's make it fluctuate around a base direction
        base_direction = 270 # Westerly
        if random.random() < 0.1: # 10% chance of a significant shift (e.g., Shamal onset/offset)
            self.current_wind_direction = random.uniform(270, 315) # North-westerly
            self.current_wind_speed = random.uniform(7.0, 15.0) # Stronger winds
        else:
            self.current_wind_direction = (base_direction + random.uniform(-15, 15)) % 360
            self.current_wind_speed = np.clip(self.current_wind_speed + random.uniform(-0.5, 0.5), 2.0, 10.0)

    def _update_temperature_humidity(self, time_step_hours):
        # Simulate diurnal temperature and humidity cycles
        self.time_of_day = (self.time_of_day + time_step_hours) % 24

        # Simple sinusoidal model for temperature
        daily_temp_swing = 5.0 # degrees C
        avg_temp = 30.0 # degrees C
        self.current_temperature = avg_temp + daily_temp_swing * np.sin(np.pi * (self.time_of_day - 8) / 12)

        # Inverse relationship for humidity (simplified)
        daily_humidity_swing = 0.1
        avg_humidity = 0.5
        self.current_humidity = avg_humidity - daily_humidity_swing * np.sin(np.pi * (self.time_of_day - 8) / 12)
        self.current_humidity = np.clip(self.current_humidity, 0.3, 0.9)

    def update(self, time_step_seconds):
        """
        Updates the meteorological conditions for the given time step.
        """
        time_step_hours = time_step_seconds / 3600.0
        self._update_wind(time_step_hours)
        self._update_temperature_humidity(time_step_hours)

    def get_current_conditions(self):
        """
        Returns the current meteorological conditions.
        """
        # Convert wind speed and direction to a vector (simplified 2D for now)
        # Direction 0 is North, 90 East, 180 South, 270 West
        # We want wind *from* a direction, so vector points *to* that direction
        angle_rad = np.deg2rad(self.current_wind_direction)
        wind_vector_x = self.current_wind_speed * np.sin(angle_rad)
        wind_vector_y = self.current_wind_speed * np.cos(angle_rad)

        return {
            "wind_speed": self.current_wind_speed,
            "wind_direction": self.current_wind_direction,
            "wind_vector": np.array([wind_vector_x, wind_vector_y, 0.0]), # Adding a Z component for consistency
            "temperature": self.current_temperature,
            "humidity": self.current_humidity,
            "time_of_day": self.time_of_day
        }

# Example usage:
if __name__ == "__main__":
    meteo = Meteorology()
    print("Initial conditions:", meteo.get_current_conditions())
    for _ in range(24): # Simulate 24 hours in 1-hour steps
        meteo.update(3600) # 1 hour step
        conditions = meteo.get_current_conditions()
        print(f"Time: {conditions['time_of_day']:.2f}h, Temp: {conditions['temperature']:.2f}°C, Humidity: {conditions['humidity']:.2f}, Wind: {conditions['wind_speed']:.2f}m/s @ {conditions['wind_direction']:.0f}°")
