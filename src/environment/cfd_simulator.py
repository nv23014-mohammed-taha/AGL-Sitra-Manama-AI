import numpy as np
import random
from scipy.ndimage import gaussian_filter
from src.environment.meteorology import Meteorology
from src.environment.urban_geometry import UrbanGeometry

class CFDSimulator:
    """
    Simulates the Computational Fluid Dynamics (CFD) environment for the AGL framework.
    This class acts as a wrapper around a conceptual OpenFOAM simulation, providing
    environmental observations to agents and processing their actions.
    """
    def __init__(self, domain_size=(10000, 5000, 500), num_residential_cells=100, num_intersections=15, grid_resolution=(50, 25, 10)):
        self.domain_size = domain_size # (length, width, height) in meters
        self.grid_resolution = grid_resolution # (grid_x, grid_y, grid_z) for PM2.5 field
        self.time_step = 60 # seconds, for internal simulation updates
        self.current_time = 0

        # Initialize 3D PM2.5 field
        self.pm25_field = np.zeros(self.grid_resolution) # Example: 50x25x10 grid
        self.wind_field = np.zeros(self.grid_resolution + (3,)) # (grid_x, grid_y, grid_z, 3) for (x,y,z) components
        self.temperature_field = np.full(self.grid_resolution, 25.0) # degrees Celsius
        self.humidity_field = np.full(self.grid_resolution, 0.6) # relative humidity

        # Integrate Meteorology and UrbanGeometry
        self.meteorology = Meteorology()
        self.urban_geometry = UrbanGeometry() # Can pass a path if needed

        # Simplified representation of residential areas and traffic intersections
        self.residential_cells = self._generate_random_locations(num_residential_cells, self.grid_resolution[:2])
        self.intersection_locations = self._generate_random_locations(num_intersections, self.grid_resolution[:2])
        self.traffic_data = {i: {
            'queue_length': random.randint(5, 50),
            'avg_speed': random.uniform(10, 40),
            'vehicle_count': random.randint(100, 1000)
        } for i in range(num_intersections)}

        # Industrial emission sources (fixed for now)
        self.industrial_sources = [
            (int(self.grid_resolution[0]*0.1), int(self.grid_resolution[1]*0.1), 0, 20), # x, y, z, emission_rate
            (int(self.grid_resolution[0]*0.2), int(self.grid_resolution[1]*0.3), 0, 15)
        ]

        print(f"CFD Simulator initialized with domain size {domain_size} and grid resolution {grid_resolution}.")

    def _generate_random_locations(self, count, grid_shape_2d):
        locations = []
        for _ in range(count):
            x = random.randint(0, grid_shape_2d[0] - 1)
            y = random.randint(0, grid_shape_2d[1] - 1)
            locations.append((x, y))
        return locations

    def _apply_temperature_inversion(self):
        """
        Simulates a temperature inversion layer, trapping pollutants below a certain height.
        """
        inversion_height_idx = int(self.grid_resolution[2] * 0.3) # Example: 30% of max height
        if random.random() < 0.3: # 30% chance of inversion
            # Make air above inversion_height_idx warmer than below
            for z in range(inversion_height_idx, self.grid_resolution[2]):
                self.temperature_field[:, :, z] = np.maximum(self.temperature_field[:, :, z], self.temperature_field[:, :, inversion_height_idx-1] + 2)
            # This would reduce vertical mixing in PM2.5 transport
            # For now, it's a conceptual flag for the transport model
            return True
        return False

    def _update_environmental_conditions(self):
        """
        Simulates the dynamic changes in PM2.5, wind, temperature, etc.
        Integrates with Meteorology for more realistic patterns.
        """
        self.meteorology.update(self.time_step)
        current_meteo = self.meteorology.get_current_conditions()

        # Update wind field based on meteorology
        main_wind_vector = current_meteo["wind_vector"]
        for z in range(self.grid_resolution[2]):
            # Simple vertical profile: wind speed increases with height
            height_factor = (z + 1) / self.grid_resolution[2]
            self.wind_field[:, :, z, 0] = main_wind_vector[0] * height_factor
            self.wind_field[:, :, z, 1] = main_wind_vector[1] * height_factor
            self.wind_field[:, :, z, 2] = main_wind_vector[2] * height_factor # Z component if any

        # Update temperature and humidity fields (simplified to be uniform for now)
        self.temperature_field.fill(current_meteo["temperature"])
        self.humidity_field.fill(current_meteo["humidity"])

        # Apply temperature inversion effect
        inversion_active = self._apply_temperature_inversion()

        # Simulate industrial emissions
        for sx, sy, sz, rate in self.industrial_sources:
            self.pm25_field[sx, sy, sz] += rate

        # Simulate vehicular emissions (simplified: add to intersection locations)
        for ix, iy in self.intersection_locations:
            # Emission rate depends on traffic queue length
            emission_rate = sum(data['queue_length'] for data in self.traffic_data.values()) / len(self.traffic_data) * 0.1
            self.pm25_field[ix, iy, 0] += emission_rate # Emissions at ground level

        # Conceptual OpenFOAM integration point:
        # In a real system, this is where OpenFOAM would be called with updated
        # wind, temperature, emissions, and BDC/STC actions to calculate PM2.5 transport.
        # For now, we use a simplified advection-diffusion model.
        self._advect_and_diffuse_pm25(inversion_active)

        self.pm25_field = np.clip(self.pm25_field, 0, 100) # Clip PM2.5 values

    def _advect_and_diffuse_pm25(self, inversion_active):
        """
        Simulates PM2.5 advection (transport by wind) and diffusion.
        """
        # Advection (wind transport)
        new_pm25_field = np.copy(self.pm25_field)
        for z in range(self.grid_resolution[2]):
            for y in range(self.grid_resolution[1]):
                for x in range(self.grid_resolution[0]):
                    wind_x, wind_y, wind_z = self.wind_field[x, y, z]

                    # Simple upstream advection (Eulerian backward differencing)
                    prev_x = int(np.clip(x - wind_x * self.time_step / 100, 0, self.grid_resolution[0] - 1))
                    prev_y = int(np.clip(y - wind_y * self.time_step / 100, 0, self.grid_resolution[1] - 1))
                    prev_z = int(np.clip(z - wind_z * self.time_step / 100, 0, self.grid_resolution[2] - 1))

                    new_pm25_field[x, y, z] = self.pm25_field[prev_x, prev_y, prev_z]

        self.pm25_field = new_pm25_field * 0.95 # Decay

        # Diffusion (Gaussian blur, potentially anisotropic based on inversion)
        sigma_z = 1.0 # Default vertical diffusion
        if inversion_active:
            sigma_z = 0.2 # Reduced vertical diffusion during inversion

        # Apply Gaussian filter in 3D
        self.pm25_field = gaussian_filter(self.pm25_field, sigma=(1, 1, sigma_z))

    def get_observation_for_filtration_agent(self, agent_position_3d):
        """
        Provides local environmental observations for a FiltrationAgent.
        agent_position_3d: (grid_x, grid_y, grid_z)
        """
        grid_x, grid_y, grid_z = int(agent_position_3d[0]), int(agent_position_3d[1]), int(agent_position_3d[2])
        grid_x = np.clip(grid_x, 0, self.grid_resolution[0]-1)
        grid_y = np.clip(grid_y, 0, self.grid_resolution[1]-1)
        grid_z = np.clip(grid_z, 0, self.grid_resolution[2]-1)

        current_meteo = self.meteorology.get_current_conditions()

        return {
            'local_pm25': self.pm25_field[grid_x, grid_y, grid_z],
            'local_wind_vector': self.wind_field[grid_x, grid_y, grid_z, :],
            'temperature': self.temperature_field[grid_x, grid_y, grid_z],
            'humidity': self.humidity_field[grid_x, grid_y, grid_z],
            'light_intensity': current_meteo["time_of_day"] / 24.0 # Simple proxy for light
        }

    def get_observation_for_traffic_agent(self, intersection_id, agent_position_2d):
        """
        Provides local traffic and environmental observations for a TrafficAgent.
        """
        grid_x, grid_y = int(agent_position_2d[0]), int(agent_position_2d[1])
        grid_x = np.clip(grid_x, 0, self.grid_resolution[0]-1)
        grid_y = np.clip(grid_y, 0, self.grid_resolution[1]-1)

        traffic_params = self.traffic_data.get(intersection_id, {})
        # For PM2.5, take ground level
        return {
            'queue_length': traffic_params.get('queue_length', 0),
            'avg_speed': traffic_params.get('avg_speed', 0),
            'vehicle_count': traffic_params.get('vehicle_count', 0),
            'local_pm25': self.pm25_field[grid_x, grid_y, 0]
        }

    def apply_filtration_action(self, agent_position_3d, action):
        """
        Applies the action of a FiltrationAgent to the environment.
        Action: {"delta_fan_speed": float, "filter_on": int, "delta_nutrient": float}
        """
        grid_x, grid_y, grid_z = int(agent_position_3d[0]), int(agent_position_3d[1]), int(agent_position_3d[2])
        grid_x = np.clip(grid_x, 0, self.grid_resolution[0]-1)
        grid_y = np.clip(grid_y, 0, self.grid_resolution[1]-1)
        grid_z = np.clip(grid_z, 0, self.grid_resolution[2]-1)

        # Simulate PM2.5 reduction based on fan speed and filter status
        # Stronger effect with higher fan speed and if filter is on
        reduction_factor = (action['delta_fan_speed'] * 0.5 + (0.5 if action['filter_on'] else 0)) * 0.1
        self.pm25_field[grid_x, grid_y, grid_z] = max(0, self.pm25_field[grid_x, grid_y, grid_z] - reduction_factor * 20)
        # Bio-activity and nutrient effects would be more complex, affecting future PM2.5 decay

    def apply_traffic_action(self, intersection_id, action):
        """
        Applies the action of a TrafficAgent to the environment.
        Action: {"delta_green_time": float, "next_phase_override": int or None}
        """
        # Simulate impact on traffic flow and indirectly on PM2.5 emissions
        current_queue = self.traffic_data[intersection_id].get('queue_length', 0)
        current_speed = self.traffic_data[intersection_id].get('avg_speed', 0)

        # Adjust queue length based on green time changes
        self.traffic_data[intersection_id]['queue_length'] = max(0, current_queue - action['delta_green_time'] * 2)
        self.traffic_data[intersection_id]['avg_speed'] = np.clip(current_speed + action['delta_green_time'] * 0.5, 5, 60)

        # Indirectly reduce PM2.5 emissions due to smoother traffic
        grid_x, grid_y = self.intersection_locations[intersection_id]
        # Reduce PM2.5 at ground level around the intersection
        self.pm25_field[grid_x, grid_y, 0] = max(0, self.pm25_field[grid_x, grid_y, 0] - action['delta_green_time'] * 0.5)

    def step(self, filtration_actions, traffic_actions):
        """
        Advances the simulation by one time step, applying all agent actions
        and updating environmental conditions.
        """
        self.current_time += self.time_step

        # Apply filtration agent actions
        for agent_id, action in filtration_actions.items():
            # Assuming agent_id maps to a position for simplicity
            # In a real setup, agents would have their own position attribute
            # For now, let's use a dummy position or pass it from agent object
            # Using a fixed z-level for BDCs for simplicity
            dummy_pos_3d = (random.randint(0, self.grid_resolution[0]-1), random.randint(0, self.grid_resolution[1]-1), 0)
            self.apply_filtration_action(dummy_pos_3d, action)

        # Apply traffic agent actions
        for intersection_id, action in traffic_actions.items():
            self.apply_traffic_action(intersection_id, action)

        # Update global environmental conditions (e.g., wind, diffusion, new emissions)
        self._update_environmental_conditions()

        # Update traffic data randomly for next step if not controlled by agents
        for i in self.traffic_data:
            self.traffic_data[i]['queue_length'] = max(0, self.traffic_data[i]['queue_length'] + random.randint(-5, 5))
            self.traffic_data[i]['avg_speed'] = np.clip(self.traffic_data[i]['avg_speed'] + random.uniform(-2, 2), 5, 60)

        return self.get_global_state()

    def get_global_state(self):
        """
        Returns the current global state of the environment.
        """
        return {
            'pm25_field': self.pm25_field,
            'wind_field': self.wind_field,
            'temperature_field': self.temperature_field,
            'humidity_field': self.humidity_field,
            'traffic_data': self.traffic_data,
            'residential_cells': self.residential_cells
        }

    def calculate_population_weighted_exposure(self, pm25_field, population_density_map=None):
        """
        Calculates the Population-Weighted Exposure (PWE) based on the PM2.5 field.
        For simplicity, assume uniform population density in residential cells if no map is provided.
        """
        if population_density_map is None:
            # Create a dummy population density map for residential cells at ground level
            population_density_map = np.zeros(pm25_field.shape[:2]) # 2D map
            for r_x, r_y in self.residential_cells:
                population_density_map[r_x, r_y] = 100 # Example density

        pwe = 0
        for r_x, r_y in self.residential_cells:
            # Assume exposure is primarily at ground level (z=0)
            pwe += pm25_field[r_x, r_y, 0] * population_density_map[r_x, r_y]
        return pwe

    def calculate_average_traffic_latency(self, traffic_data):
        """
        Calculates the average traffic latency across all intersections.
        """
        total_latency = 0
        num_intersections = len(traffic_data)
        if num_intersections == 0: return 0

        # Simplified latency calculation: higher queue length means higher latency
        for data in traffic_data.values():
            total_latency += data.get('queue_length', 0) * 0.1 # Convert queue to latency in minutes
        return total_latency / num_intersections

    def calculate_energy_consumption(self, filtration_actions):
        """
        Calculates the energy consumption of BDCs based on their actions.
        """
        total_energy = 0
        for action in filtration_actions.values():
            # Higher fan speed and filter_on consume more energy
            total_energy += action['delta_fan_speed'] * 5 + (10 if action['filter_on'] else 0)
        return total_energy
