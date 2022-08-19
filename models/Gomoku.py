import numpy as np
import torch
from torch import nn, optim
from models.res_unit import ResUnit
from typing import Literal
from models.Super_model import Model

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

        
class Gomoku(Model):
    def step(self, prev_state: np.ndarray, action: int, player: Literal[-1, 1]):
        return_state = np.copy(prev_state)
        return_state[action // 15, action % 15] += player

        return return_state
    
    def action_space(self):
        return 225
    
    def valid_actions(self, cur_state: np.ndarray):
        for i in range(15):
            for j in range(15):
                if cur_state[i, j] == 0:
                    yield i * 15 + j
    
    def determine_winner(self, cur_state: np.ndarray):
        for i in range(15):
            for j in range(11):
                hor = np.sum(cur_state[i, j:j+5])
                ver = np.sum(cur_state[j:j+5, i])

                if hor == 5 or ver == 5:
                    return 1
                if hor == -5 or ver == -5:
                    return -1

        for i in range(11):
            for j in range(11):
                dia0 = np.sum(cur_state[range(i, i+5), range(i, i+5)])
                dia1 = np.sum(cur_state[range(i, i+5), range(i+4, i-1, -1)])

                if dia0 == 5 or dia1 == 5:
                    return 1
                if dia0 == -5 or dia1 == -5:
                    return -1

        return 0

    def init_state(self):
        return np.zeros((15, 15), dtype = np.int8)

