import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np
import random
import os

from src.agents.filtration_agent import FiltrationAgent
from src.agents.traffic_agent import TrafficAgent
from src.agents.coordinator import Coordinator
from src.environment.cfd_simulator import CFDSimulator
from src.training.reward_function import MultiObjectiveReward
from src.utils.config import Config
from src.utils.visualization import AGLVisualizer

class Actor(nn.Module):
    """
    Actor network for an individual agent in MAPPO.
    Outputs logits for discrete actions and mean/std for continuous actions.
    """
    def __init__(self, obs_dim, discrete_action_dim, continuous_action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)

        # Discrete action head
        self.discrete_head = nn.Linear(128, discrete_action_dim)

        # Continuous action head (mean and log_std)
        self.continuous_mean_head = nn.Linear(128, continuous_action_dim)
        self.continuous_log_std_head = nn.Linear(128, continuous_action_dim)

        # Initialize log_std to be small
        self.continuous_log_std_head.weight.data.fill_(0.0)
        self.continuous_log_std_head.bias.data.fill_(-0.5)

    def forward(self, obs):
        x = torch.relu(self.fc1(obs))
        x = torch.relu(self.fc2(x))

        discrete_logits = self.discrete_head(x)
        continuous_mean = torch.tanh(self.continuous_mean_head(x)) # Scale to [-1, 1]
        continuous_log_std = self.continuous_log_std_head(x)
        continuous_std = torch.exp(continuous_log_std)

        return discrete_logits, continuous_mean, continuous_std

class Critic(nn.Module):
    """
    Critic network for MAPPO (centralized critic).
    Takes global state and outputs a value estimate.
    """
    def __init__(self, global_state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(global_state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1) # Output a single value estimate

    def forward(self, global_state):
        x = torch.relu(self.fc1(global_state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class MAPPOTrainer:
    """
    Implements the Multi-Agent Proximal Policy Optimization (MAPPO) training pipeline.
    Uses Centralized Training with Decentralized Execution (CTDE).
    """
    def __init__(self, config):
        self.config = config
        self.gamma = config.GAMMA
        self.gae_lambda = config.GAE_LAMBDA
        self.clip_epsilon = config.CLIP_EPSILON
        self.entropy_coeff = config.ENTROPY_COEFF
        self.num_epochs = config.PPO_NUM_EPOCHS # Number of PPO epochs per update

        # Initialize environment and reward calculator
        self.cfd_simulator = CFDSimulator(
            domain_size=config.DOMAIN_SIZE,
            num_residential_cells=config.NUM_RESIDENTIAL_CELLS,
            num_intersections=config.NUM_INTERSECTIONS,
            grid_resolution=config.GRID_RESOLUTION
        )
        self.reward_calculator = MultiObjectiveReward(
            alpha=config.REWARD_ALPHA,
            beta=config.REWARD_BETA,
            gamma=config.REWARD_GAMMA
        )
        self.coordinator = Coordinator(
            num_filtration_agents=config.NUM_FILTRATION_AGENTS,
            num_traffic_agents=config.NUM_TRAFFIC_AGENTS
        )
        self.visualizer = AGLVisualizer(results_dir=config.RESULTS_DIR)

        # Determine global state dimension from a sample state
        sample_global_state = self.cfd_simulator.get_global_state()
        self.global_state_dim = self.coordinator.get_global_observation_for_critic(sample_global_state).shape[0]

        # Initialize agents with dummy positions for now
        self.filtration_agents = [
            FiltrationAgent(i, position=(random.randint(0, config.GRID_RESOLUTION[0]-1), random.randint(0, config.GRID_RESOLUTION[1]-1), 0))
            for i in range(config.NUM_FILTRATION_AGENTS)
        ]
        self.traffic_agents = [
            TrafficAgent(i, intersection_id=i)
            for i in range(config.NUM_INTERSECTIONS)
        ]

        # Create actor networks for each agent type
        # Filtration Agent: 1 discrete action (filter_on: 2 states), 2 continuous (delta_fan_speed, delta_nutrient)
        self.filtration_actors = [
            Actor(config.FILTRATION_OBS_DIM, config.FILTRATION_ACTION_DIM_DISCRETE, config.FILTRATION_ACTION_DIM_CONTINUOUS)
            for _ in range(config.NUM_FILTRATION_AGENTS)
        ]
        # Traffic Agent: 1 discrete action (next_phase_override: 4 states), 1 continuous (delta_green_time)
        self.traffic_actors = [
            Actor(config.TRAFFIC_OBS_DIM, config.TRAFFIC_ACTION_DIM_DISCRETE, config.TRAFFIC_ACTION_DIM_CONTINUOUS)
            for _ in range(config.NUM_TRAFFIC_AGENTS)
        ]

        # Create a centralized critic network
        self.critic = Critic(self.global_state_dim)

        # Optimizers
        self.actor_optimizers = [
            optim.Adam(actor.parameters(), lr=config.LEARNING_RATE)
            for actor in self.filtration_actors + self.traffic_actors
        ]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.LEARNING_RATE)

        print("MAPPO Trainer initialized.")
        print(f"Global state dimension for critic: {self.global_state_dim}")

    def _obs_dict_to_tensor(self, obs_dict, obs_dim):
        """
        Converts an observation dictionary into a flattened tensor.
        This needs to be consistent with how the observation space is defined.
        """
        if obs_dim == self.config.FILTRATION_OBS_DIM:
            tensor_data = [
                obs_dict.get("local_pm25", 0.0),
                obs_dict.get("local_wind_vector", [0.0, 0.0, 0.0])[0],
                obs_dict.get("local_wind_vector", [0.0, 0.0, 0.0])[1],
                obs_dict.get("local_wind_vector", [0.0, 0.0, 0.0])[2],
                obs_dict.get("temperature", 0.0),
                obs_dict.get("humidity", 0.0),
                obs_dict.get("light_intensity", 0.0),
                obs_dict.get("fan_speed", 0.0) # Assuming fan_speed is part of observation for agent itself
            ]
        elif obs_dim == self.config.TRAFFIC_OBS_DIM:
            tensor_data = [
                obs_dict.get("queue_length", 0.0),
                obs_dict.get("avg_speed", 0.0),
                obs_dict.get("vehicle_count", 0.0),
                obs_dict.get("local_pm25", 0.0),
                obs_dict.get("current_signal_phase", 0.0)
            ]
        else:
            raise ValueError(f"Unknown observation dimension: {obs_dim}")

        return torch.tensor(tensor_data, dtype=torch.float32)

    def _compute_gae(self, rewards, values, dones, next_value):
        """
        Computes Generalized Advantage Estimation (GAE).
        """
        advantages = []
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_v = next_value
            else:
                next_v = values[t + 1]
            delta = rewards[t] + self.gamma * next_v * (1 - dones[t]) - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae_lam
            advantages.insert(0, last_gae_lam)
        return torch.tensor(advantages, dtype=torch.float32)

    def train(self, num_episodes=None):
        num_episodes = num_episodes if num_episodes is not None else self.config.NUM_EPISODES
        print(f"Starting MAPPO training for {num_episodes} episodes...")

        # Histories for plotting
        all_episode_rewards = []
        all_avg_pm25 = []
        all_avg_latency = []

        for episode in range(num_episodes):
            current_global_state = self.cfd_simulator.get_global_state()
            done = False
            episode_rewards = []
            episode_states = []
            episode_values = []
            episode_dones = []

            # Buffers for collecting trajectory data for PPO update
            all_obs = {agent.agent_id: [] for agent in self.filtration_agents + self.traffic_agents}
            all_actions_discrete_log_probs = {agent.agent_id: [] for agent in self.filtration_agents + self.traffic_agents}
            all_actions_continuous_log_probs = {agent.agent_id: [] for agent in self.filtration_agents + self.traffic_agents}
            all_actions_continuous_means = {agent.agent_id: [] for agent in self.filtration_agents + self.traffic_agents}
            all_actions_continuous_stds = {agent.agent_id: [] for agent in self.filtration_agents + self.traffic_agents}
            all_actions_taken_discrete = {agent.agent_id: [] for agent in self.filtration_agents + self.traffic_agents}
            all_actions_taken_continuous = {agent.agent_id: [] for agent in self.filtration_agents + self.traffic_agents}

            steps_in_episode = 0

            while not done and steps_in_episode < self.config.MAX_STEPS_PER_EPISODE:
                # Get global state for critic
                global_state_tensor = torch.tensor(self.coordinator.get_global_observation_for_critic(current_global_state), dtype=torch.float32)
                value = self.critic(global_state_tensor).squeeze(0).item()
                episode_states.append(global_state_tensor)
                episode_values.append(value)

                filtration_agent_actions_for_env = {}
                traffic_agent_actions_for_env = {}

                # Filtration Agents
                for agent in self.filtration_agents:
                    dummy_pos_3d = agent.position # Use agent's actual position
                    obs_dict = self.cfd_simulator.get_observation_for_filtration_agent(dummy_pos_3d)
                    obs_tensor = self._obs_dict_to_tensor(obs_dict, self.config.FILTRATION_OBS_DIM)
                    all_obs[agent.agent_id].append(obs_tensor)

                    discrete_logits, continuous_mean, continuous_std = self.filtration_actors[agent.agent_id](obs_tensor)

                    # Discrete action (filter_on)
                    dist_discrete = Categorical(logits=discrete_logits)
                    action_discrete = dist_discrete.sample()
                    all_actions_discrete_log_probs[agent.agent_id].append(dist_discrete.log_prob(action_discrete))
                    all_actions_taken_discrete[agent.agent_id].append(action_discrete.item())

                    # Continuous actions (delta_fan_speed, delta_nutrient)
                    dist_continuous = Normal(continuous_mean, continuous_std)
                    action_continuous = dist_continuous.sample()
                    all_actions_continuous_log_probs[agent.agent_id].append(dist_continuous.log_prob(action_continuous).sum())
                    all_actions_continuous_means[agent.agent_id].append(continuous_mean)
                    all_actions_continuous_stds[agent.agent_id].append(continuous_std)
                    all_actions_taken_continuous[agent.agent_id].append(action_continuous.detach().numpy())

                    # Scale continuous actions for environment
                    delta_fan_speed = action_continuous[0].item() * 0.2 # Scale to [-0.2, 0.2]
                    delta_nutrient = action_continuous[1].item() * 0.1 # Scale to [-0.1, 0.1]

                    action_for_env = {
                        'delta_fan_speed': delta_fan_speed,
                        'filter_on': action_discrete.item(),
                        'delta_nutrient': delta_nutrient
                    }
                    filtration_agent_actions_for_env[agent.agent_id] = action_for_env

                # Traffic Agents
                for agent in self.traffic_agents:
                    dummy_pos_2d = self.cfd_simulator.intersection_locations[agent.intersection_id]
                    obs_dict = self.cfd_simulator.get_observation_for_traffic_agent(agent.intersection_id, dummy_pos_2d)
                    obs_tensor = self._obs_dict_to_tensor(obs_dict, self.config.TRAFFIC_OBS_DIM)
                    all_obs[agent.agent_id].append(obs_tensor)

                    discrete_logits, continuous_mean, continuous_std = self.traffic_actors[agent.agent_id](obs_tensor)

                    # Discrete action (next_phase_override)
                    dist_discrete = Categorical(logits=discrete_logits)
                    action_discrete = dist_discrete.sample()
                    all_actions_discrete_log_probs[agent.agent_id].append(dist_discrete.log_prob(action_discrete))
                    all_actions_taken_discrete[agent.agent_id].append(action_discrete.item())

                    # Continuous action (delta_green_time)
                    dist_continuous = Normal(continuous_mean, continuous_std)
                    action_continuous = dist_continuous.sample()
                    all_actions_continuous_log_probs[agent.agent_id].append(dist_continuous.log_prob(action_continuous).sum())
                    all_actions_continuous_means[agent.agent_id].append(continuous_mean)
                    all_actions_continuous_stds[agent.agent_id].append(continuous_std)
                    all_actions_taken_continuous[agent.agent_id].append(action_continuous.detach().numpy())

                    # Scale continuous actions for environment
                    delta_green_time = action_continuous[0].item() * 10 # Scale to [-10, 10]

                    action_for_env = {
                        'delta_green_time': delta_green_time,
                        'next_phase_override': action_discrete.item() if random.random() < 0.5 else None # Randomly apply override
                    }
                    traffic_agent_actions_for_env[agent.agent_id] = action_for_env

                # Environment step
                next_global_state = self.cfd_simulator.step(filtration_agent_actions_for_env, traffic_agent_actions_for_env)

                # Calculate reward
                reward = self.reward_calculator.calculate_reward(
                    current_global_state, next_global_state,
                    filtration_agent_actions_for_env, traffic_agent_actions_for_env,
                    self.cfd_simulator
                )
                episode_rewards.append(reward)
                episode_dones.append(False) # Assuming episode doesn't end mid-step

                current_global_state = next_global_state
                steps_in_episode += 1

                # Visualize PM2.5 heatmap periodically
                if steps_in_episode % self.config.VISUALIZATION_INTERVAL == 0 or steps_in_episode == self.config.MAX_STEPS_PER_EPISODE:
                    # Get 2D ground level PM2.5 for heatmap
                    pm25_2d = current_global_state['pm25_field'][:, :, 0]
                    # Get 2D positions for agents
                    filt_agent_pos_2d = [(int(a.position[0]), int(a.position[1])) for a in self.filtration_agents]
                    traffic_agent_loc_2d = self.cfd_simulator.intersection_locations
                    self.visualizer.plot_pm25_heatmap(
                        pm25_2d,
                        self.cfd_simulator.residential_cells,
                        filt_agent_pos_2d,
                        traffic_agent_loc_2d,
                        step_num=episode * self.config.MAX_STEPS_PER_EPISODE + steps_in_episode
                    )

            # Episode ends, last value for GAE
            next_global_state_tensor = torch.tensor(self.coordinator.get_global_observation_for_critic(current_global_state), dtype=torch.float32)
            next_value = self.critic(next_global_state_tensor).squeeze(0).item()
            episode_dones[-1] = True # Mark last step as done for GAE calculation

            # Compute GAE advantages and returns
            advantages = self._compute_gae(episode_rewards, episode_values, episode_dones, next_value)
            returns = advantages + torch.tensor(episode_values, dtype=torch.float32)

            # --- PPO Update Phase ---
            # Prepare data for batching
            all_agent_ids = list(all_obs.keys())
            # Flatten lists of tensors for batching
            batch_obs = {aid: torch.cat(all_obs[aid]) for aid in all_agent_ids}
            batch_actions_discrete_log_probs = {aid: torch.cat(all_actions_discrete_log_probs[aid]) for aid in all_agent_ids}
            batch_actions_continuous_log_probs = {aid: torch.cat(all_actions_continuous_log_probs[aid]) for aid in all_agent_ids}
            batch_actions_taken_discrete = {aid: torch.tensor(all_actions_taken_discrete[aid], dtype=torch.long) for aid in all_agent_ids}
            batch_actions_taken_continuous = {aid: torch.tensor(np.array(all_actions_taken_continuous[aid]), dtype=torch.float32) for aid in all_agent_ids}

            batch_global_states = torch.stack(episode_states)
            batch_returns = returns
            batch_advantages = advantages

            # Normalize advantages
            batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

            for _ in range(self.num_epochs):
                # Critic update
                values_pred = self.critic(batch_global_states).squeeze(-1)
                critic_loss = nn.MSELoss()(values_pred, batch_returns)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

                # Actor update for each agent
                for i, agent in enumerate(self.filtration_agents + self.traffic_agents):
                    agent_id = agent.agent_id
                    current_actor = self.filtration_actors[i] if isinstance(agent, FiltrationAgent) else self.traffic_actors[i - len(self.filtration_agents)]

                    discrete_logits, continuous_mean, continuous_std = current_actor(batch_obs[agent_id])

                    # Discrete PPO loss
                    dist_discrete = Categorical(logits=discrete_logits)
                    new_log_probs_discrete = dist_discrete.log_prob(batch_actions_taken_discrete[agent_id])
                    ratio_discrete = torch.exp(new_log_probs_discrete - batch_actions_discrete_log_probs[agent_id])
                    surr1_discrete = ratio_discrete * batch_advantages
                    surr2_discrete = torch.clamp(ratio_discrete, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                    actor_loss_discrete = -torch.min(surr1_discrete, surr2_discrete).mean()
                    entropy_discrete = dist_discrete.entropy().mean()

                    # Continuous PPO loss
                    dist_continuous = Normal(continuous_mean, continuous_std)
                    new_log_probs_continuous = dist_continuous.log_prob(batch_actions_taken_continuous[agent_id]).sum(axis=-1)
                    ratio_continuous = torch.exp(new_log_probs_continuous - batch_actions_continuous_log_probs[agent_id])
                    surr1_continuous = ratio_continuous * batch_advantages
                    surr2_continuous = torch.clamp(ratio_continuous, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                    actor_loss_continuous = -torch.min(surr1_continuous, surr2_continuous).mean()
                    entropy_continuous = dist_continuous.entropy().mean()

                    # Combined actor loss
                    actor_loss = actor_loss_discrete + actor_loss_continuous - self.entropy_coeff * (entropy_discrete + entropy_continuous)

                    self.actor_optimizers[i].zero_grad()
                    actor_loss.backward(retain_graph=True if i < len(self.filtration_agents) + len(self.traffic_agents) - 1 else False)
                    self.actor_optimizers[i].step()

            # Log metrics for plotting
            all_episode_rewards.append(sum(episode_rewards))
            all_avg_pm25.append(self.cfd_simulator.calculate_population_weighted_exposure(current_global_state['pm25_field']) / self.config.NUM_RESIDENTIAL_CELLS)
            all_avg_latency.append(self.cfd_simulator.calculate_average_traffic_latency(current_global_state['traffic_data']))

            if episode % 10 == 0:
                print(f"Episode {episode}/{num_episodes}, Total Reward: {sum(episode_rewards):.2f}, Critic Loss: {critic_loss.item():.4f}")

        print("MAPPO training complete.")

        # Generate final training plots
        self.visualizer.plot_training_progress(all_episode_rewards, all_avg_pm25, all_avg_latency)
        self.visualizer.plot_pm25_time_series(all_avg_pm25, title="Average Residential PM2.5 Over Episodes", filename="avg_pm25_over_episodes.png")
        self.visualizer.plot_traffic_latency_time_series(all_avg_latency, title="Average Traffic Latency Over Episodes", filename="avg_traffic_latency_over_episodes.png")

    def save_models(self, path):
        """
        Saves the trained actor and critic models.
        """
        torch.save({
            'filtration_actors_state_dicts': [actor.state_dict() for actor in self.filtration_actors],
            'traffic_actors_state_dicts': [actor.state_dict() for actor in self.traffic_actors],
            'critic_state_dict': self.critic.state_dict(),
        }, path)
        print(f"Models saved to {path}")

    def load_models(self, path):
        """
        Loads trained actor and critic models.
        """
        checkpoint = torch.load(path)
        for i, actor_state_dict in enumerate(checkpoint['filtration_actors_state_dicts']):
            self.filtration_actors[i].load_state_dict(actor_state_dict)
        for i, actor_state_dict in enumerate(checkpoint['traffic_actors_state_dicts']):
            self.traffic_actors[i].load_state_dict(actor_state_dict)
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        print(f"Models loaded from {path}")

# Example usage (for testing purposes)
if __name__ == "__main__":
    config = Config()
    config.print_config()

    trainer = MAPPOTrainer(config)
    trainer.train(num_episodes=50) # Run a small number of episodes for testing
    trainer.save_models(config.MODEL_SAVE_PATH)

    # To load and continue training or evaluate:
    # new_trainer = MAPPOTrainer(config)
    # new_trainer.load_models(config.MODEL_SAVE_PATH)
    # new_trainer.train(num_episodes=50) # Continue training
