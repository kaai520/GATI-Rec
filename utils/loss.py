import torch
import torch.nn as nn
import torch.nn.functional as F


class MCLosses(object):
    def __init__(self):
        pass

    def build_loss(self, mode='mse'):
        if mode == 'mse':
            return self.MseLoss

    def MseLoss(self, pred, target):
        loss = F.mse_loss(pred, target)
        return loss


if __name__ == '__main__':
    input = torch.rand(10)
    gt = torch.rand(10)
    loss = MCLosses().build_loss()
    print(loss(input, gt))


