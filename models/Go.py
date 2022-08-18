from models.res_unit import ResUnit
from torch import nn 

class Go_network(nn.Module):
    def __init__(self):
        self.input = ResUnit(5, 256)
        self.tower = nn.ModuleList(ResUnit() for _ in range(39))
        self.flatten = nn.Flatten()
        self.actor = nn.Sequential(
            nn.Linear(256*19*19, 1024),
            nn.ReLU(),
            nn.Linear(1024, 19*19),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(256*19*19, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input(x)
        for layer in self.tower:
            x = layer(x)
        x = self.flatten(x)
        policy = self.actor(x)
        value = self.critic(x)

        return policy, value

class Go:
    pass