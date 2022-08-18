import numpy as np

class Model:
    def step(self, prev_state: np.ndarray, action: int, player):
        raise NotImplementedError
    
    def action_space(self):
        raise NotImplementedError

    def valid_actions(self, cur_state: np.ndarray):
        raise NotImplementedError

    def determine_winner(self, cur_state: np.ndarray):
        raise NotImplementedError
    
    def init_state(self):
        raise NotImplementedError

    