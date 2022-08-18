import numpy as np
import torch
from torch import nn, optim
from res_unit import ResUnit
from typing import Literal

class ACnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = ResUnit(1, 128, 3, 1)
        self.tower = nn.ModuleList(ResUnit(128, 128) for _ in range(4))
        self.flatten = nn.Flatten(-3, -1)
        self.actor = nn.Sequential(
            nn.Linear(15 * 15 * 128, 15 * 15),
            nn.Softmax(dim=-1),
        )
        self.critic = nn.Sequential(
            nn.Linear(15 * 15 * 128, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.input(x)
        for layer in self.tower:
            x = layer(x)
        
        x = self.flatten(x)
        probs = self.actor(x)
        value = self.critic(x)

        return probs, value

        
class Gomoku:
    pass