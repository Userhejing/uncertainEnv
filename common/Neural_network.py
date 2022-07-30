import torch
import torch.nn as nn
from typing import List, Type

def Mlp_modules(
    neuron_nums: List,
    activation_fn: Type[nn.Module] = nn.ReLU
) -> List:
    assert len(neuron_nums) >= 2, 'the number of neuron is too small. '
    modules = []
    for i in range(len(neuron_nums) - 1):
        modules.append(nn.Linear(neuron_nums[i], neuron_nums[i + 1]))
        if i != len(neuron_nums) - 2:
            modules.append(activation_fn())
    return modules

class Mlp_Network(nn.Module):
    def __init__(
        self,
        neuron_nums: List,
        activation_fn: Type[nn.Module] = nn.ReLU,
        classfier: bool = False,
    ) -> None:
        """
        Construct a mlp based on parameters.
        :param neuron_nums: a list containing neuron numbers of each layer.
        :param activation_fn: activation function type. [except for the final layer which depends on the "classfier"]
        :param classfier: represents whether to use softmax at the last layer.
            By default False, aften used for Q_network to calculate Q value. [logits]
            If True, often used to calculate probability of each action. [probs]
        """
        super(Mlp_Network, self).__init__()
        modules = Mlp_modules(neuron_nums, activation_fn)
        if classfier:
            modules.append(nn.Softmax(dim=1))
        self.architecture = nn.Sequential(
            *modules
        )
    
    def forward(self, obs):
        return self.architecture(obs)
