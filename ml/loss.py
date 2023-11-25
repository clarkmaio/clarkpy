
import numpy as np
from typing import Iterable, Union

import torch
from torch.nn import Module,MSELoss, ModuleList, L1Loss



def loss_hub(loss: Union[str, Module]):
    '''Function to parse loss function in both case Module and string'''
    if isinstance(loss, str):
        if loss.lower() == 'crps':
            return CRPSLoss()
        elif loss.lower() in ('mseloss', 'mse'):
            return MSELoss()
        elif loss.lower() in ('maeloss', 'mae'):
            return L1Loss()
        else:
            raise ValueError('Loss function not implemented')
    else:
        return loss

# Pinball Loss Function
class PinballLoss(Module):
    def __init__(self, tau):
        super(PinballLoss, self).__init__()
        self.tau = tau

    def forward(self, y_pred, y):
        delta = y_pred - y
        return torch.mean(torch.max(self.tau*delta, (self.tau-1)*delta))

# Compute loss function as average of a set of PinPallLoss functions
class AveragePinballLoss(Module):
    '''
    Loss = 1/N sum_q PinballLoss(tau=q)(y_pred, y)
    '''
    def __init__(self, quantiles: Iterable[float]):
        self.quantiles = quantiles
        super(AveragePinballLoss, self).__init__()
        self.losses = ModuleList([PinballLoss(tau) for tau in quantiles])

    def forward(self, y_pred, y):
        loss = torch.stack([self.losses[i](y_pred[:, [i]], y) for i in range(len(self.quantiles))])
        return torch.mean(loss)


class CRPSLoss(Module):
    def __init__(self):
        super(CRPSLoss, self).__init__()

    def forward(self, y_pred, y):
        # Compute 1/n * sum_S [ 1/2m^2 * sum_i sum_j |y_pred[S, i] - y_pred[S, j]| ]
        mix_loss = self.mix_loss(y_pred)

        # Compute 1/n * sum_S [ sum_i |y_pred[S, i] - y[S]| ]
        mae_loss = self.mae_loss(y_pred, y)

        return mae_loss - mix_loss

    def mae_loss(self, y_pred, y):
        return torch.mean(torch.abs(y_pred - y))

    def mix_loss(self, y_pred):
        n, m = y_pred.size()
        # expanded_ens = y_pred.unsqueeze(1).unsqueeze(3)
        # expanded_ens_transposed = expanded_ens.permute(0, 3, 2, 1)
        # mix_loss = torch.sum(torch.abs(expanded_ens - expanded_ens_transposed))
        # mix_loss = mix_loss / (2 * m ** 2)

        # Calcolo della seconda parte della formula
        mix_loss = torch.sum(torch.abs(y_pred.unsqueeze(1) - y_pred.unsqueeze(2))) / (n * 2 * m ** 2)

        return mix_loss