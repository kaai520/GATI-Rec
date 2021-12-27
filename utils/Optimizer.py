import torch


class Optimizer(object):
    def __init__(self, lr = 0.01, weight_decay = 0.0005, momentum=0.9, mode = None):
        self.lr = lr
        self.mode = mode
        self.weight_decay = weight_decay
        self.momentum = momentum

    def build_optim(self, train_params):
        if self.mode == 'Adam':
            return torch.optim.Adam(train_params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.mode == 'SGD':
            return torch.optim.SGD(train_params, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        elif self.mode == 'RMSprop':
            return torch.optim.RMSprop(train_params, lr=self.lr, weight_decay=self.weight_decay, momentum=self.momentum)
        elif self.mode == 'Adamax':
            return torch.optim.Adamax(train_params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.mode == 'AdamW':
            return torch.optim.AdamW(train_params, lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError
