import numpy as np
import torch
import torch.nn as nn
from typing import List


class Sine(nn.Module):
    r"""Applies the sine function with frequency scaling element-wise:

    :math:`\text{Sine}(x)= \sin(\omega * x)`

    Args:
        omega: factor used for scaling the frequency

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """

    def __init__(self, omega):
        super().__init__()
        self.omega = omega

    def forward(self, x):
        return torch.sin(self.omega * x)

def make_module(module):
    # Create a module instance if we don't already have one
    if isinstance(module, torch.nn.Module):
        return module
    else:
        return module()


class FullyConnectedBlock(torch.nn.Module):
    def __init__(self, dim_in, dim_out, bias=True, activation=torch.nn.ReLU):
        super().__init__()

        self.linear = torch.nn.Linear(dim_in, dim_out, bias=bias)
        self.activation = make_module(activation) if activation is not None else torch.nn.Identity()

    def forward(self, input):
        return self.activation(self.linear(input))

class FullyConnectedResidualBlock(torch.nn.Module):
    def __init__(self, dim_in, dims_hidden, dim_out, bias=True,
                 activation_hidden=torch.nn.ReLU, activation=torch.nn.ReLU):
        super().__init__()

        self.dimensions = [dim_in] + dims_hidden + [dim_out]
        self.num_layers = len(self.dimensions) - 1

        # The only reason why we add the residual layers explicitly to this module
        # instead of using nn.Sequential, is that the graph visualization looks better
        # (e. g. in Tensorboard)
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                layer = FullyConnectedBlock(self.dimensions[i], self.dimensions[i + 1], activation=None)
            else:
                layer = FullyConnectedBlock(self.dimensions[i], self.dimensions[i + 1], activation=make_module(activation_hidden))
            self.add_module(f'Residual{i:d}', layer)

        self.shortcut = torch.nn.Identity() if dim_in == dim_out else torch.nn.Linear(dim_in, dim_out) 

        self.activation = torch.nn.Identity() if activation is None else make_module(activation)

    def forward(self, input):
        Fx = input
        for i in range(self.num_layers):
            Fx = self.__getattr__(f'Residual{i:d}')(Fx)

        x = self.shortcut(input)

        return self.activation(Fx + x)


def siren_init_first(**kwargs):
    module = kwargs['module']
    n = kwargs['n']
    if isinstance(module, nn.Linear):
        module.weight.data.uniform_(-1 / n, 
                                     1 / n)

def siren_init(**kwargs):
    module = kwargs['module']
    n = kwargs['n']
    omega = kwargs['omega']
    if isinstance(module, nn.Linear):
        module.weight.data.uniform_(-np.sqrt(6 / n) / omega, 
                                     np.sqrt(6 / n) / omega)

def init_weights_normal(**kwargs):
    module = kwargs['module']
    if isinstance(module, nn.Linear):
        if hasattr(module, 'weight'):
            nn.init.kaiming_normal_(module.weight, a=0.0, nonlinearity='relu', mode='fan_in')
        if hasattr(module, 'bias'):
            nn.init.zeros_(module.bias)

def init_weights_normal_last(**kwargs):
    module = kwargs['module']
    if isinstance(module, nn.Linear):
        if hasattr(module, 'weight'):
            nn.init.xavier_normal_(module.weight, gain=1)
            module.weight.data = -torch.abs(module.weight.data)
        if hasattr(module, 'bias'):
            nn.init.zeros_(module.bias)
            
class FC(nn.Module):
    def __init__(self, in_features, out_features, hidden_features: List[int], activation='relu', last_activation=None, bias=True, first_omega=30, hidden_omega=30.0):
        super().__init__()

        layers = []

        activations_and_inits = {
            'sine': (Sine(first_omega),
                     siren_init,
                     siren_init_first,
                     None),
            'relu': (nn.ReLU(inplace=True),
                     init_weights_normal,
                     init_weights_normal,
                     None),
            'relu2': (nn.ReLU(inplace=True),
                     init_weights_normal,
                     init_weights_normal,
                     init_weights_normal_last),
            'softplus': (nn.Softplus(),
                        init_weights_normal,
                        None)
        }

        activation_fn, weight_init, first_layer_init, last_layer_init = activations_and_inits[activation]


        # First layer
        layer = FullyConnectedBlock(in_features, hidden_features[0], bias=bias, activation=activation_fn)
        if first_layer_init is not None: 
            layer.apply(lambda module: first_layer_init(module=module, n=in_features))
        layers.append(layer)

        for i in range(len(hidden_features)):
            n = hidden_features[i]

            # Initialize the layer right away
            layer = FullyConnectedBlock(n, n, bias=bias, activation=activation_fn)
            layer.apply(lambda module: weight_init(module=module, n=n, omega=hidden_omega))
            layers.append(layer)

        # Last layer
        layer = FullyConnectedBlock(hidden_features[-1], out_features, bias=bias, activation=last_activation)
        layer.apply(lambda module: weight_init(module=module, n=hidden_features[-1], omega=hidden_omega))
        if last_layer_init is not None: 
            layer.apply(lambda module: last_layer_init(module=module, n=in_features))
        layers.append(layer)

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
