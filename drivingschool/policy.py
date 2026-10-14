import math
import torch
from torch import nn
from .env import OBS_SIZE


class Policy(nn.Module):
    def __init__(self, width=256):
        super().__init__()
        self.width = width
        self.body = nn.Sequential(nn.Linear(OBS_SIZE, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.actor = nn.Linear(width, 2)
        self.critic = nn.Linear(width, 1)
        self.logstd = nn.Parameter(torch.tensor([-0.5, -0.5]))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, math.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight, 0.01)
        nn.init.orthogonal_(self.critic.weight, 1)

    def forward(self, obs):
        hidden = self.body(obs)
        return self.actor(hidden), self.critic(hidden).squeeze(-1)

    def sample(self, obs):
        mean, value = self(obs)
        std = self.logstd.clamp(-3, 0.5).exp()
        action = mean + std * torch.randn_like(mean)
        logprob = (-0.5 * ((action-mean)/std)**2 - std.log() - 0.5 * math.log(2*math.pi)).sum(-1)
        return action, logprob, value

    def score(self, obs, action):
        mean, value = self(obs)
        logstd = self.logstd.clamp(-3, 0.5)
        std = logstd.exp()
        logprob = (-0.5 * ((action-mean)/std)**2 - logstd - 0.5 * math.log(2*math.pi)).sum(-1)
        entropy = (logstd + 0.5 * math.log(2*math.pi*math.e)).sum()
        return logprob, entropy, value
