from MCTS import MCTS
import torch
from torch.nn import functional
from collections import deque
import random

class AlphaZero:
    def __init__(self, model, network, device = 'cuda', take_turns: bool = True):
        self.model = model
        self.network = network
        self.network.to(device)
        self.take_turns = take_turns
        self.device = device
        self.optimizer = torch.optim.Adam(self.network.parameters())
        self.stack = deque()

    def play(self, episode_count = 10):
        mcts = MCTS(self.model, self.network, self.take_turns)
        for _ in range(episode_count):
            self.stack.extend(mcts.run())
    
    def _tensor(self, x):
        return torch.tensor(x, dtype = torch.float32, device = self.device)

    def train(self, batch_size = 64, epoch = 8):
        for _ in range(epoch):
            batch = random.sample(self.stack, batch_size)
            states, policies, vals = zip(*batch)

            states = torch.stack(tuple(map(self._tensor, states)))
            policies = torch.stack(tuple(map(self._tensor, policies)))
            vals = self._tensor(vals).unsqueeze(-1)

            out_policies, out_vals = self.network(states)

            loss = functional.mse_loss(out_vals, vals) + functional.cross_entropy(out_policies, policies)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


