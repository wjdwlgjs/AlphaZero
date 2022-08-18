import torch
from torch import nn
import numpy as np
from typing import Literal
from models.Super_model import Model

class Tictactoe_ACnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(-2, -1),
            nn.Linear(9, 64),
            nn.ReLU()
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 1),
            nn.Tanh()
        )
        self.actor = nn.Sequential(
            nn.Linear(64, 9),
            nn.Softmax(dim =-1)
        )
    
    def forward(self, x):
        x = self.network(x)
        probs = self.actor(x)
        value = self.critic(x)
        
        return probs, value


class TicTacToe(Model):
    def step(self, prev_state: np.ndarray, action: int, player: Literal[-1, 1]):
        new_state = np.copy(prev_state)
        new_state[action // 3, action % 3] += player
        return new_state
    
    def valid_actions(self, cur_state):
        for i in range(3):
            for j in range(3):
                if int(cur_state[i, j]) == 0:
                    yield i * 3 + j
    
    def determine_winner(self, cur_state):
        if np.sum(np.abs(cur_state)) == 9:
            return 0

        for i in range(3):
            ver = np.sum(cur_state[:, i])
            hor = np.sum(cur_state[i, :])
        
            if ver == 3 or hor == 3:
                return 1
            if ver == -3 or hor == -3:
                return -1
            
        dia0 = np.sum(cur_state[range(3), range(3)]) 
        dia1 = np.sum(cur_state[range(3), range(2, -1, -1)])

        if dia0 == 3 or dia1 == 3:
            return 1
        if dia0 == -3 or dia1 == -3:
            return -1
        
        return None

    def action_space(self):
        return 9

    def init_state(self):
        return np.zeros((3, 3), dtype = np.int8)
    
    
            