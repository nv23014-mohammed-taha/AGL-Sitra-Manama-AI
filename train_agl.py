import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from agl_core import UrbanEnvironment
from datetime import datetime
import json


class RolloutBuffer:
    def __init__(self, n_agents, gamma=0.99, gae_lambda=0.95):
        self.n_agents = n_agents
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.reset()

    def reset(self):
        self.obs = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store(self, obs, actions, log_probs, rewards, values, dones):
        self.obs.append(obs)
        self.actions.append(actions)
        self.log_probs.append(log_probs)
        self.rewards.append(rewards)
        self.values.append(values)
        self.dones.append(dones)

    def compute(self, next_values):
        T = len(self.rewards)
        adv = np.zeros((T, self.n_agents))
        ret = np.zeros((T, self.n_agents))

        for i in range(self.n_agents):
            gae = 0
            for t in reversed(range(T)):
                next_v = next_values[i] if t == T - 1 else self.values[t + 1][i]
                delta = self.rewards[t][i] + self.gamma * next_v * (1 - self.dones[t][i]) - self.values[t][i]
                gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[t][i]) * gae
                adv[t, i] = gae

        ret = adv + np.array(self.values)
        return adv, ret


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, act_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        mean = torch.tanh(self.net(x))
        std = torch.exp(self.log_std)
        return mean, std


class CentralCritic(nn.Module):
    def __init__(self, global_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        return self.net(x)


class MAPPOAgent:
    def __init__(self, obs_dim, global_dim, act_dim):
        self.actor = Actor(obs_dim, act_dim)
        self.critic = CentralCritic(global_dim)
        self.opt_actor = optim.Adam(self.actor.parameters(), lr=3e-4)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=3e-4)

    def act(self, obs):
        obs = torch.FloatTensor(obs)
        mean, std = self.actor(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum()
        return action.detach().numpy(), logp.detach().numpy()

    def value(self, global_obs):
        return self.critic(torch.FloatTensor(global_obs)).detach().numpy()


def train():
    env = UrbanEnvironment()
    n_agents = env.n_agents

    obs_dim = len(env.reset()[0])
    global_dim = obs_dim * n_agents
    act_dim = 2

    agents = [MAPPOAgent(obs_dim, global_dim, act_dim) for _ in range(n_agents)]
    buffer = RolloutBuffer(n_agents)

    T = 200000
    t = 0

    while t < T:
        obs = env.reset()
        done = False

        while not done:
            actions = {}
            logps = {}
            vals = {}

            global_state = np.concatenate(list(obs.values()))

            for i in range(n_agents):
                a, lp = agents[i].act(obs[i])
                v = agents[i].value(global_state)[0]
                actions[i] = a
                logps[i] = lp
                vals[i] = v

            next_obs, rewards, dones, _ = env.step(actions)

            buffer.store(
                obs,
                actions,
                logps,
                rewards,
                vals,
                dones
            )

            obs = next_obs
            t += 1
            done = all(dones.values())

        next_global = np.concatenate(list(obs.values()))
        next_vals = [agents[i].value(next_global)[0] for i in range(n_agents)]

        adv, ret = buffer.compute(next_vals)

        for i in range(n_agents):
            obs_batch = torch.FloatTensor(np.array([b[i] for b in buffer.obs]))
            act_batch = torch.FloatTensor(np.array([b[i] for b in buffer.actions]))
            old_logp = torch.FloatTensor(np.array([b[i] for b in buffer.log_probs]))
            adv_batch = torch.FloatTensor(adv[:, i])
            ret_batch = torch.FloatTensor(ret[:, i])

            mean, std = agents[i].actor(obs_batch)
            dist = torch.distributions.Normal(mean, std)

            new_logp = dist.log_prob(act_batch).sum(dim=1)

            ratio = torch.exp(new_logp - old_logp)
            clip = torch.clamp(ratio, 0.8, 1.2)

            loss_actor = -(torch.min(ratio * adv_batch, clip * adv_batch)).mean()
            loss_critic = ((agents[i].critic(torch.FloatTensor(np.array(buffer.obs).reshape(len(buffer.obs), -1))) - ret_batch) ** 2).mean()

            agents[i].opt_actor.zero_grad()
            loss_actor.backward()
            agents[i].opt_actor.step()

            agents[i].opt_critic.zero_grad()
            loss_critic.backward()
            agents[i].opt_critic.step()

        buffer.reset()

        if t % 5000 == 0:
            print("step", t)

    torch.save([a.actor.state_dict() for a in agents], "mappo_actor.pt")

    return agents


def evaluate(env, agents):
    rewards = []
    for _ in range(50):
        obs = env.reset()
        done = False
        r = 0

        while not done:
            actions = {}
            global_state = np.concatenate(list(obs.values()))

            for i in range(len(agents)):
                actions[i], _ = agents[i].act(obs[i])

            obs, rew, done, _ = env.step(actions)
            r += np.mean(list(rew.values()))
            done = all(done.values())

        rewards.append(r)

    return {"avg_reward": float(np.mean(rewards))}


if __name__ == "__main__":
    env = UrbanEnvironment()
    agents = train()
    print(evaluate(env, agents))
