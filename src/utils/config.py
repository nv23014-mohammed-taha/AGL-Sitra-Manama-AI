class Config:
    """
    Global configuration settings for the Autonomous Green Lung (AGL) project.
    This class centralizes parameters for agents, environment, and training.
    """

    # Environment Settings
    DOMAIN_SIZE = (10000, 5000, 500)  # (length, width, height) in meters
    GRID_RESOLUTION = (50, 25, 10)    # (grid_x, grid_y, grid_z) for PM2.5 field
    SIMULATION_TIME_STEP = 60         # seconds per simulation step
    NUM_RESIDENTIAL_CELLS = 100       # Number of conceptual residential areas
    NUM_INTERSECTIONS = 15            # Number of traffic intersections

    # Agent Settings
    NUM_FILTRATION_AGENTS = 20
    NUM_TRAFFIC_AGENTS = 15

    # Filtration Agent specific
    FILTRATION_OBS_DIM = 8  # PM2.5, wind_x, wind_y, wind_z, temp, humidity, light, fan_speed
    FILTRATION_ACTION_DIM_DISCRETE = 2 # filter_on (binary)
    FILTRATION_ACTION_DIM_CONTINUOUS = 2 # delta_fan_speed, delta_nutrient

    # Traffic Agent specific
    TRAFFIC_OBS_DIM = 5     # queue_length, avg_speed, vehicle_count, local_pm25, current_signal_phase
    TRAFFIC_ACTION_DIM_DISCRETE = 4 # next_phase_override (4 phases)
    TRAFFIC_ACTION_DIM_CONTINUOUS = 1 # delta_green_time

    # MARL Training Settings (PPO specific)
    LEARNING_RATE = 3e-4
    GAMMA = 0.99              # Discount factor for future rewards
    GAE_LAMBDA = 0.95         # GAE parameter
    CLIP_EPSILON = 0.2        # PPO clipping parameter
    ENTROPY_COEFF = 0.01      # Entropy regularization coefficient
    PPO_NUM_EPOCHS = 10       # Number of PPO epochs per update
    NUM_EPISODES = 5000       # Total training episodes
    MAX_STEPS_PER_EPISODE = 200 # Max steps before an episode terminates
    BATCH_SIZE = 64           # Batch size for policy updates (for mini-batch PPO, not fully implemented yet)
    VISUALIZATION_INTERVAL = 50 # Visualize every N steps

    # Reward Function Weights
    REWARD_ALPHA = 0.6        # Weight for Population-Weighted Exposure (PWE)
    REWARD_BETA = 0.3         # Weight for Traffic Latency Deviation (TLD)
    REWARD_GAMMA = 0.1        # Weight for Energy Consumption (EC)

    # Paths
    MODEL_SAVE_PATH = "checkpoints/trained_agents.pth"
    LOG_DIR = "logs/tensorboard"
    RESULTS_DIR = "results/"

    # Simulation Specifics (for CFD)
    OPENFOAM_PATH = "/opt/OpenFOAM/OpenFOAM-v2212" # Example path
    CFD_RESOLUTION_FACTOR = 1.0 # Factor to scale CFD grid resolution

    def __init__(self):
        pass

    @classmethod
    def print_config(cls):
        print("\n--- AGL Configuration ---")
        for attr, value in vars(cls).items():
            if not attr.startswith("__") and not callable(value):
                print(f"{attr}: {value}")
        print("-------------------------")

# Example usage:
if __name__ == "__main__":
    Config.print_config()
