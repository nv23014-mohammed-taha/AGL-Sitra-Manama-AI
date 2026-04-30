import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

class AGLVisualizer:
    """
    Provides visualization utilities for the Autonomous Green Lung (AGL) project.
    Generates heatmaps of PM2.5, agent positions, and training progress plots.
    """
    def __init__(self, results_dir="results/"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        print(f"AGL Visualizer initialized. Results will be saved to {self.results_dir}")

    def plot_pm25_heatmap(self, pm25_field_2d, residential_cells, filtration_agent_positions_2d, traffic_agent_intersection_locations_2d, step_num, title="PM2.5 Concentration Heatmap"):
        """
        Generates a 2D heatmap of PM2.5 concentrations, overlaying residential areas and agent positions.
        pm25_field_2d: A 2D numpy array representing PM2.5 concentrations (e.g., ground level).
        residential_cells: List of (x, y) tuples for residential areas.
        filtration_agent_positions_2d: List of (x, y) tuples for filtration agents.
        traffic_agent_intersection_locations_2d: List of (x, y) tuples for traffic agents.
        step_num: Current simulation step number for filename.
        """
        plt.figure(figsize=(12, 8))
        sns.heatmap(pm25_field_2d.T, cmap="viridis", cbar_kws={"label": "PM2.5 Concentration (µg/m³)"})
        plt.title(f"{title} - Step {step_num}")
        plt.xlabel("Grid X")
        plt.ylabel("Grid Y")

        # Overlay residential cells
        res_x = [cell[0] for cell in residential_cells]
        res_y = [cell[1] for cell in residential_cells]
        plt.scatter(res_x, res_y, marker="s", color="red", s=50, label="Residential Areas", alpha=0.7)

        # Overlay filtration agents
        filt_x = [pos[0] for pos in filtration_agent_positions_2d]
        filt_y = [pos[1] for pos in filtration_agent_positions_2d]
        plt.scatter(filt_x, filt_y, marker="o", color="blue", s=100, label="Filtration Agents", alpha=0.8)

        # Overlay traffic agents
        traffic_x = [loc[0] for loc in traffic_agent_intersection_locations_2d]
        traffic_y = [loc[1] for loc in traffic_agent_intersection_locations_2d]
        plt.scatter(traffic_x, traffic_y, marker="X", color="orange", s=100, label="Traffic Agents", alpha=0.8)

        plt.legend()
        plt.gca().invert_yaxis() # Make (0,0) bottom-left
        filename = os.path.join(self.results_dir, f"pm25_heatmap_step_{step_num:05d}.png")
        plt.savefig(filename)
        plt.close()
        return filename

    def plot_training_progress(self, episode_rewards, avg_pm25_history, avg_latency_history, filename="training_progress.png"):
        """
        Plots the training progress over episodes.
        """
        fig, axs = plt.subplots(3, 1, figsize=(12, 18))

        # Plot Episode Rewards
        axs[0].plot(episode_rewards)
        axs[0].set_title("Episode Rewards Over Training")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Total Reward")
        axs[0].grid(True)

        # Plot Average PM2.5 Concentration
        axs[1].plot(avg_pm25_history)
        axs[1].set_title("Average Residential PM2.5 Concentration Over Training")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Avg PM2.5 (µg/m³)")
        axs[1].grid(True)

        # Plot Average Traffic Latency
        axs[2].plot(avg_latency_history)
        axs[2].set_title("Average Traffic Latency Over Training")
        axs[2].set_xlabel("Episode")
        axs[2].set_ylabel("Avg Latency (minutes)")
        axs[2].grid(True)

        plt.tight_layout()
        filename = os.path.join(self.results_dir, filename)
        plt.savefig(filename)
        plt.close()
        return filename

    def plot_pm25_distribution(self, pm25_field, step_num, title="PM2.5 Distribution Histogram"):
        """
        Plots a histogram of PM2.5 concentrations across the field.
        """
        plt.figure(figsize=(10, 6))
        plt.hist(pm25_field.flatten(), bins=50, color=\
"blue", alpha=0.7)
        plt.title(f"{title} - Step {step_num}")
        plt.xlabel("PM2.5 Concentration (µg/m³)")
        plt.ylabel("Frequency")
        plt.grid(True)
        filename = os.path.join(self.results_dir, f"pm25_distribution_step_{step_num:05d}.png")
        plt.savefig(filename)
        plt.close()
        return filename

    def plot_pm25_time_series(self, pm25_history, title="Average PM2.5 Over Time", filename="pm25_time_series.png"):
        """
        Plots the average PM2.5 concentration over simulation steps or episodes.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(pm25_history)
        plt.title(title)
        plt.xlabel("Time Step / Episode")
        plt.ylabel("Average PM2.5 (µg/m³)")
        plt.grid(True)
        filename = os.path.join(self.results_dir, filename)
        plt.savefig(filename)
        plt.close()
        return filename

    def plot_traffic_latency_time_series(self, latency_history, title="Average Traffic Latency Over Time", filename="traffic_latency_time_series.png"):
        """
        Plots the average traffic latency over simulation steps or episodes.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(latency_history)
        plt.title(title)
        plt.xlabel("Time Step / Episode")
        plt.ylabel("Average Latency (minutes)")
        plt.grid(True)
        filename = os.path.join(self.results_dir, filename)
        plt.savefig(filename)
        plt.close()
        return filename
