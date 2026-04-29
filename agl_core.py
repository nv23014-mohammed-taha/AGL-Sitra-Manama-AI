import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import deque


@dataclass
class AgentConfig:
    agent_id: int
    agent_type: str
    observation_dim: int
    action_dim: int
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5


class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        x = self.net(x)
        mean = torch.tanh(self.mean(x))
        std = torch.exp(self.log_std)
        return mean, std


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class MARLAgent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(config.observation_dim, config.action_dim, config.hidden_dim).to(self.device)
        self.value = ValueNetwork(config.observation_dim, config.hidden_dim).to(self.device)

        self.policy_opt = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.value_opt = optim.Adam(self.value.parameters(), lr=config.learning_rate)

        self.memory = {
            "obs": deque(maxlen=2048),
            "act": deque(maxlen=2048),
            "rew": deque(maxlen=2048),
            "val": deque(maxlen=2048),
            "logp": deque(maxlen=2048),
            "done": deque(maxlen=2048)
        }

    def select_action(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mean, std = self.policy(obs)
            value = self.value(obs)

        dist = Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(dim=1)

        return (
            action.cpu().numpy()[0],
            float(logp.cpu().numpy()[0]),
            float(value.cpu().numpy()[0])
        )

    def store(self, obs, act, rew, val, logp, done):
        self.memory["obs"].append(obs)
        self.memory["act"].append(act)
        self.memory["rew"].append(rew)
        self.memory["val"].append(val)
        self.memory["logp"].append(logp)
        self.memory["done"].append(done)

    def advantage(self, next_value):
        r = np.array(self.memory["rew"])
        v = np.array(self.memory["val"])
        d = np.array(self.memory["done"])

        adv = np.zeros_like(r)
        gae = 0

        for t in reversed(range(len(r))):
            nv = next_value if t == len(r) - 1 else v[t + 1]
            delta = r[t] + self.config.gamma * nv * (1 - d[t]) - v[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - d[t]) * gae
            adv[t] = gae

        ret = adv + v
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def update(self, adv, ret):
        obs = torch.tensor(np.array(self.memory["obs"]), dtype=torch.float32).to(self.device)
        act = torch.tensor(np.array(self.memory["act"]), dtype=torch.float32).to(self.device)
        old_logp = torch.tensor(np.array(self.memory["logp"]), dtype=torch.float32).to(self.device)
        adv = torch.tensor(adv, dtype=torch.float32).to(self.device)
        ret = torch.tensor(ret, dtype=torch.float32).to(self.device)

        for _ in range(3):
            mean, std = self.policy(obs)
            dist = Normal(mean, std)
            logp = dist.log_prob(act).sum(dim=1)

            ratio = torch.exp(logp - old_logp)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 0.8, 1.2) * adv

            policy_loss = -torch.min(s1, s2).mean()
            entropy = dist.entropy().mean()

            loss_pi = policy_loss - self.config.entropy_coef * entropy
            self.policy_opt.zero_grad()
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.policy_opt.step()

            value = self.value(obs).squeeze()
            loss_v = ((value - ret) ** 2).mean()

            self.value_opt.zero_grad()
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(self.value.parameters(), self.config.max_grad_norm)
            self.value_opt.step()

        self.memory = {k: deque(maxlen=2048) for k in self.memory}


class UrbanEnvironment:
    def __init__(self, n_filtration_agents=20, n_traffic_agents=15, domain_size=(10000, 5000, 500), grid_resolution=10):
        self.n_f = n_filtration_agents
        self.n_t = n_traffic_agents
        self.n_agents = self.n_f + self.n_t

        self.domain_size = domain_size
        self.grid_resolution = grid_resolution
        self.grid_shape = tuple(s // grid_resolution for s in domain_size)

        self.pm25 = np.random.uniform(20, 35, self.grid_shape)
        self.wind = np.random.uniform(-5, 5, (*self.grid_shape, 2))
        self.traffic = np.random.uniform(0.3, 0.9, self.n_t)

        self.bdc = [(np.random.randint(0, self.grid_shape[0]),
                     np.random.randint(0, self.grid_shape[1]),
                     np.random.randint(0, self.grid_shape[2])) for _ in range(self.n_f)]

        self.stc = [(np.random.randint(0, self.grid_shape[0]),
                     np.random.randint(0, self.grid_shape[1])) for _ in range(self.n_t)]

        self.t = 0
        self.max_t = 10000

    def obs_dim(self):
        return len(self.get_obs(0))

    def get_obs(self, i):
        if i < self.n_f:
            x, y, z = self.bdc[i]
            pm = self.pm25[max(0, x-2):x+3, max(0, y-2):y+3, max(0, z-1):z+2].flatten()
            wd = self.wind[max(0, x-2):x+3, max(0, y-2):y+3, max(0, z-1):z+2].flatten()
            return np.concatenate([pm, wd, [self.pm25[x, y, z]]]).astype(np.float32)
        x, y = self.stc[i - self.n_f]
        pm = self.pm25[max(0, x-2):x+3, max(0, y-2):y+3].flatten()
        tr = self.traffic[max(0, i-2):i+3]
        return np.concatenate([pm, tr, [self.traffic[i - self.n_f]]]).astype(np.float32)

    def step(self, actions):
        self.t += 1

        for i, a in actions.items():
            if i < self.n_f:
                x, y, z = self.bdc[i]
                self.pm25[x, y, z] *= (1 - ((a[0] + 1) / 2) * ((a[1] + 1) / 2) * 0.3)
            else:
                j = i - self.n_f
                self.traffic[j] = np.clip(self.traffic[j] + (a[0] * 0.1), 0.1, 0.95)

        self.pm25 += np.random.normal(0, 0.5, self.grid_shape)
        self.pm25 = np.clip(self.pm25, 10, 100)

        obs = {i: self.get_obs(i) for i in range(self.n_agents)}

        pm = np.mean(self.pm25)
        tr = np.mean(self.traffic)

        rew = {i: (-pm / 50.0 if i < self.n_f else -(tr - 0.5) ** 2) for i in range(self.n_agents)}
        done = {i: self.t >= self.max_t for i in range(self.n_agents)}

        return obs, rew, done


class MARLEngine:
    def __init__(self, env, n_f=20, n_t=15):
        self.env = env
        self.n_agents = env.n_agents

        obs_dim = env.obs_dim()

        self.agents = {
            i: MARLAgent(AgentConfig(i, "f" if i < n_f else "t", obs_dim, 2))
            for i in range(self.n_agents)
        }

    def train(self, steps=100000):
        t = 0

        while t < steps:
            obs = self.env.step({})[0]
            done = False

            ep_done = False

            while not ep_done:
                actions = {}
                vals = {}

                for i in range(self.n_agents):
                    a, lp, v = self.agents[i].select_action(obs[i])
                    actions[i] = a
                    vals[i] = v

                next_obs, rew, done = self.env.step(actions)

                for i in range(self.n_agents):
                    self.agents[i].store(obs[i], actions[i], rew[i], vals[i], 0, done[i])

                obs = next_obs
                t += 1
                ep_done = all(done.values())

            for i in range(self.n_agents):
                adv, ret = self.agents[i].advantage(0.0)
                self.agents[i].update(adv, ret)
