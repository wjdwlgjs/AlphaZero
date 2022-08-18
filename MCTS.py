import numpy as np
import torch
from models.Super_model import Model
from typing import Literal
from collections import deque

class Node:
    def __init__(self, prior: float, state: np.ndarray, prev_action, whose_turn: Literal[-1, 1], model: Model, network, take_turns: bool = True, parent = None):
        self.prior = prior
        self.state = state
        self.player = whose_turn
        self.model = model
        self.network = network
        self.parent = parent
        self.take_turns = take_turns
        self.prev_action = prev_action
        
        self.children = []
        self.value_sum = 0
        self.expanded = False
        self.winner = model.determine_winner(self.state)
        self.visit_count = 0
        self.action_space = model.action_space()
        

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count 

    def ucb_score(self):
        return self.value() * (1 - 2 * int(self.take_turns)) + (self.parent.visit_count) ** 0.5 / (self.visit_count + 1) * self.prior
        
    def expand(self):
        with torch.no_grad():
            probs, value = self.network(torch.tensor(self.state, dtype = torch.float32, device = 'cuda'))
            probs = probs.squeeze().cpu().numpy()

        probs = probs[list(self.model.valid_actions(self.state))]
        probs = np.exp(probs)
        probs /= np.sum(probs)
        children_player = self.player * (1 - 2 * int(self.take_turns))

        for idx, action in enumerate(self.model.valid_actions(self.state)):
            new_state = self.model.step(self.state, action, self.player)
            self.children.append(Node(probs[idx], new_state, action, children_player, self.model, self.network, self.take_turns, self))

        self.expanded = True
        return float(value)

    def traverse(self):
        if self.winner != None:
            value_plus = self.player * self.winner
            self.value_sum += value_plus
            self.visit_count += 1
            return value_plus
        
        if not self.expanded:
            value_plus = self.expand()
            self.value_sum += value_plus
            self.visit_count += 1
            return value_plus
        
        ucb_scores = np.array([child.ucb_score() for child in self.children])
        value_plus = self.children[np.argmax(ucb_scores)].traverse() * (1 - 2 * int(self.take_turns))

        self.value_sum += value_plus
        self.visit_count += 1
        return value_plus

    def clone_as_root(self):
        return Node(1, self.state, None, self.player, self.model, self.network, self.take_turns, None)

    def __repr__(self):
        return 'prev_action: {}, visit_count: {}, value_sum: {}'.format(self.prev_action, self.visit_count, self.value_sum)

class MCTS:
    def __init__(self, model: Model, network, take_turns: bool = True):
        self.model = model
        self.network = network
        self.take_turns = take_turns
        

        self.init_state = self.model.init_state()
    
    def run(self, temperature = 0):
        stack = deque()
        return_stack = deque()
        root = Node(1, self.init_state, None, 1, self.model, self.network, self.take_turns, None)

        while root.winner == None:
            for _ in range(100):
                root.traverse()

            tree_policies = np.zeros((self.model.action_space()), dtype = np.float32)
            valid_space_policies = []

            for child in root.children:
                tree_policies[child.prev_action] = child.visit_count
                valid_space_policies.append(child.visit_count)

            tree_policies /= root.visit_count
            stack.append((root.state, tree_policies, root.player))

            if temperature == 0:
                next_node = root.children[np.argmax(valid_space_policies)]
                root = next_node.clone_as_root()

        while len(stack) != 0:
            temp = stack.pop()
            return_stack.append((temp[0], temp[1], root.winner * temp[2]))

        return return_stack