import os
import torch
from tensorboardX import SummaryWriter


class torch_saver(object):
    def __init__(self,root):
        self.root = root
        self.make_dir()

    def __call__(self, state, epoch):
        filename = os.path.join(self.root, 'checkpoint_epoch{:0>3d}.pth.tar'.format(epoch))
        torch.save(state, filename)

    def make_dir(self):
        if not os.path.exists(self.root):
            os.makedirs(self.root)


def get_writer(path='log'):
    if not os.path.exists(path):
        os.makedirs(path)
    return SummaryWriter(log_dir=path)


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info('abc')

