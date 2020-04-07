import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, layer_size=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            layer_size (int): Controls the size of the hidden layerrs in the network.
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # The model will take as input a state and return the action list value as output for that given state
        self.layer1 = nn.Linear(state_size, layer_size)
        self.layer2 = nn.Linear(layer_size, layer_size)
        self.layer3 = nn.Linear(layer_size, action_size)
        
    def forward(self, state):
        """Build a network that maps state -> action values.
        The network makes use of gelu activation units between the layers.
        
        """
        
        a1 = self.layer1(state)
        h1 = F.relu(a1)
        a2 = self.layer2(h1)
        h2 = F.relu(a2)
        action_values = self.layer3(h2)
        return action_values
        
