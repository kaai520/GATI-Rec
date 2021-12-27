import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """
        reset all parameters

        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        update parameters
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SumMeter(object):
    """Computes and stores the sum and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        """
        reset all parameters
        """
        self.val_sum = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val_sum, n=1):
        """
        update parameters
        """
        self.val_sum = val_sum
        self.sum += val_sum
        self.count += n
        self.avg = self.sum / self.count


class Evaluator(object):
    def __init__(self):
        # self.mae_meter = SumMeter()
        self.mse_meter = SumMeter()

    # def mae(self):
    #     return self.mae_meter.avg

    def mse(self):
        return self.mse_meter.avg

    def rmse(self):
        return np.sqrt(self.mse_meter.avg)

    def reset(self):
        # self.mae_meter.reset()
        self.mse_meter.reset()

    def add_batch(self, gt_arr, pre_arr):
        assert gt_arr.shape == pre_arr.shape
        diff_arr = gt_arr-pre_arr
        # self.mae_meter.update(np.abs(diff_arr).sum(), n=diff_arr.size)
        self.mse_meter.update(np.square(diff_arr).sum(), n=diff_arr.size)


if __name__ == '__main__':
    evaluator = Evaluator()
    for i in range(10):
        gt_arr = np.random.rand(64, 30)
        pre_arr = np.random.rand(64, 30)
        evaluator.add_batch(gt_arr, pre_arr)
        print(evaluator.rmse())

    print('final:', evaluator.rmse())


