import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
import random
import numpy as np
from tqdm import tqdm
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn.data_parallel import DataParallel
import scipy.sparse as sp
import argparse
import warnings
warnings.simplefilter('ignore', sp.SparseEfficiencyWarning)

from modeling.GATIRec import GATIRec
from utils.logger import Logger, get_nowtime, load_yaml
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import AverageMeter, Evaluator
from utils.Optimizer import Optimizer
from utils.saver import torch_saver, get_writer
from dataloaders.datasets.MovieLens import DynamicMovieLens
from dataloaders.datasets.Flixster import Flixster
from dataloaders.datasets.Douban import Douban
from dataloaders.datasets.YahooMusic import YahooMusic
from utils.loss import MCLosses


def setup_seed(seed):
    if seed != 'None':
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


class Trainer(object):
    def __init__(self, config_name):
        self.log_dir = ''
        self.config = self._init_config(config_name)
        self.logger = self._init_log()
        self.writer = get_writer(self.log_dir)  # tensorboardX
        self.saver = torch_saver(os.path.join(self.log_dir, 'checkpoint')) #checkpoint saver
        self.device = 'cpu'
        self.cuda = self._init_gpu() # if cuda, device:'cpu'->'cuda'
        
        self.criterion = MCLosses().build_loss(self.config['loss_mode'])
        self.train_loader = self._get_dataloader('train')
        self.val_loader = self._get_dataloader('test')
        self._fix_seed()
        self.model = self._init_model()
        self.optimizer = self._init_optimizer()
        self.evaluator = Evaluator()
        self.scheduler = self._get_lr_scheduler()
        self.resume_epoch = 0
        self.train_loss = AverageMeter()
        self.val_loss = AverageMeter()
        self.val_rmse = np.inf

    def training(self, epoch):
        self.train_loss.reset()
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_sample_tr = len(self.train_loader)
        for i, sample in enumerate(tbar):
            data = sample
            if self.cuda and len(self.config['gpu_ids']) > 1:
                data_y = torch.cat([single_data.y for single_data in data]).to(self.device)
            else:
                data.to(self.device, non_blocking=True)
                data_y = data.y
            self.scheduler(self.optimizer, i, epoch)
            self.optimizer.zero_grad()
            preds = self.model(data)
            loss = self.criterion(preds, data_y)
            loss.backward()
            self.optimizer.step()
            self.train_loss.update(loss.item())
            tbar.set_description('[%d]Train loss: %.5f' % (epoch, self.train_loss.avg))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_sample_tr * epoch)
        #     if i%10 == 0:
        #         torch.cuda.empty_cache()
        # torch.cuda.empty_cache()
        self.writer.add_scalar('train/total_loss_epoch', self.train_loss.sum, epoch)
        self.writer.add_scalar('train/lr_epoch', self.optimizer.param_groups[0]['lr'], epoch)
        log_string = '[Epoch: %d, numSamples: %d] lr: %.6f Train total loss: %.4f' \
                     % (epoch, len(self.train_loader.dataset),
                        self.optimizer.param_groups[0]['lr'], self.train_loss.sum)
        self.logger.log(log_string)
        if (epoch+1) % 10 == 0:
            self.save_model(epoch+1)

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        self.val_loss.reset()
        tbar = tqdm(self.val_loader)
        with torch.no_grad():
            for i, sample in enumerate(tbar):
                data = sample
                if self.cuda and len(self.config['gpu_ids']) > 1:
                    data_y = torch.cat([single_data.y for single_data in data]).to(self.device)
                else:
                    data.to(self.device, non_blocking=True)
                    data_y = data.y
                preds = self.model(data)
                loss = self.criterion(preds, data_y)
                self.val_loss.update(loss.item())
                tbar.set_description('[%d]Val loss: %.5f' % (epoch, self.val_loss.avg))
                self.evaluator.add_batch(preds.cpu().numpy(), data_y.cpu().numpy())
            # torch.cuda.empty_cache()
            rmse = self.evaluator.rmse()
            self.writer.add_scalar('val/total_loss_epoch', self.val_loss.sum, epoch)
            self.writer.add_scalar('val/rmse', rmse, epoch)
            log_string = '[Epoch: %d, numSamples: %d] val total loss: %.3f rmse: %f'\
                         % (epoch, len(self.val_loader.dataset), self.val_loss.sum, rmse)
            self.logger.log(log_string)
            if rmse < self.val_rmse:
                self.val_rmse = rmse
                self.save_best_model()

    def save_model(self, epoch):
        # if self.config['is_cuda']:
        #     state_dict = self.model.module.state_dict()
        # else:
        state_dict = self.model.state_dict()
        self.saver({'state_dict': state_dict, 'optimizer': self.optimizer.state_dict(), }, epoch)

    def save_best_model(self):
        # if self.cuda:
        #     state_dict = self.model.module.state_dict()
        # else:
        state_dict = self.model.state_dict()
        torch.save({
            'state_dict': state_dict
        }, os.path.join(self.log_dir, 'best_model.pth.tar'))

    def _init_config(self, config_name):
        config = load_yaml('config/{}.yaml'.format(config_name))
        return config

    def _init_log(self):
        log_prefix = self.config['log_prefix']
        # dataset_root = self.config['dataset_root']
        # dataset = list(filter(lambda x: x != '', dataset_root.split('/')))[-1]
        dataset = self.config['dataset']
        momentum = self.config['momentum']
        weightdecay = self.config['weight_decay']
        lr = self.config['lr']
        bs = self.config['batch_size']
        now_time = get_nowtime()
        log_name = 'log/{}-{}-{}-momentum{}-weightdecay{}-lr{}-bs{}'.format(
            now_time, log_prefix, dataset, momentum, weightdecay, lr, bs
        )

        if not os.path.exists(log_name):
            os.makedirs(log_name)
        self.log_dir = log_name
        logger = Logger(os.path.join(self.log_dir, 'train.log'))
        logger.log('log_time:{}'.format(now_time))
        logger.log('-----------------------config-----------------------')
        for k, v in self.config.items():
            logger.log('{}={}'.format(k, v))
        logger.log('-----------------------begin-----------------------')
        return logger

    def _fix_seed(self):
        setup_seed(self.config['seed'])

    def _init_gpu(self):
        cuda = torch.cuda.is_available() and self.config['is_cuda']
        if cuda:
            self.device = 'cuda:0'
            # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = ','.join([str(x) for x in self.config['gpu_ids']])
        return cuda

    def _init_model(self):
        input_embedding = self.config['input_embedding']
        edge_classes = self.config['edge_classes']
        heads = self.config['heads']
        layers = self.config['layers']
        input_channels = self.config['input_channels'] if input_embedding else 4
        attention = True if self.config['attention'] is None else self.config['attention']
        edge_feature = True if self.config['edge_feature'] is None else self.config['edge_feature']
        try:
            use_feature = False if self.config['use_feature'] is None else self.config['use_feature']
        except:
            use_feature = False
        context_dim = None
        item_num_embeddings = None
        user_num_embeddings = None
        edge_embedding = 'cumsum' if self.config['edge_embedding'] is None else self.config['edge_embedding']
        if use_feature:
            context_dim = self.config['context_dim']
            item_num_embeddings = self.config['item_num_embeddings']
            user_num_embeddings = self.config['user_num_embeddings']
        model = GATIRec(input_channels=input_channels, EGAT_heads=heads, EGAT_output_channels=32, EGAT_layers=layers,
                        edge_classes=edge_classes, multiply_by=1, activation='elu', decoder_choice='mlp',
                        concat_nodes_feature=True, edge_embedding=edge_embedding, add_self_feature=True,
                        input_embedding=input_embedding, attention=attention, edge_feature=edge_feature,
                        use_feature=use_feature, context_dim=context_dim, item_num_embeddings=item_num_embeddings,
                        user_num_embeddings=user_num_embeddings, h=self.config['max_hops'])
        if self.config['dataset'] == 'yahoo_music':
            model.set_multiply_by(0.05)
        if self.config['dataset'] == 'flixster':
            model.set_multiply_by(0.5)
        if self.cuda:
            device_ids = [i for i in range(len(self.config['gpu_ids']))]
            if len(device_ids) > 1:
                model = DataParallel(model, device_ids=device_ids)
        model = model.to(self.device)
        return model

    def _init_optimizer(self):
        lr = self.config['lr']
        weight_decay = self.config['weight_decay']
        momentum = self.config['momentum']
        mode = self.config['optim_mode']
        params = self.model.parameters() # multi-gpus
        optimizer = Optimizer(lr=lr,
                              weight_decay=weight_decay,
                              momentum=momentum,
                              mode=mode).build_optim(params)
        return optimizer

    def _get_dataloader(self, split='train'):
        if split == 'train':
            num_workers = self.config['train_num_workers']
            shuffle = True
        else:
            num_workers = self.config['test_num_workers']
            shuffle = False
        one_hot_flag = not self.config['input_embedding']
        cluster_sample = True if self.config['cluster_sample'] is None else self.config['cluster_sample']
        if self.config['dataset'] == 'flixster':
            dataset = Flixster(root=self.config['dataset_root'],
                               max_neighbors=self.config['max_neighbors'], h=self.config['max_hops'], split=split, one_hot_flag=one_hot_flag,
                               cluster_sample=cluster_sample)
        elif self.config['dataset'] == 'douban':
            dataset = Douban(root=self.config['dataset_root'], max_neighbors=self.config['max_neighbors'], h=self.config['max_hops'],
                             split=split, one_hot_flag=one_hot_flag, cluster_sample=cluster_sample)
        elif self.config['dataset'] == 'yahoo_music':
            dataset = YahooMusic(root=self.config['dataset_root'], max_neighbors=self.config['max_neighbors'], h=self.config['max_hops'],
                                 split=split, one_hot_flag=one_hot_flag, cluster_sample=cluster_sample)
        else:
            use_feature = False if self.config['use_feature'] is None else self.config['use_feature']
            dataset = DynamicMovieLens(root=self.config['dataset_root'], dataset=self.config['dataset'],
                                       max_neighbors=self.config['max_neighbors'], h=self.config['max_hops'],
                                       split=split, one_hot_flag=one_hot_flag, use_feature=use_feature,
                                       cluster_sample=cluster_sample)

        if self.cuda and len(self.config['gpu_ids']) > 1:
            dataloader = DataListLoader(dataset, batch_size=self.config['batch_size'], shuffle=shuffle,
                                        num_workers=num_workers, pin_memory=False)
        else:
            dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=shuffle,
                                num_workers=num_workers, pin_memory=False)
        return dataloader

    def _get_lr_scheduler(self):
        mode = self.config['lr_mode']
        lr_step = 0
        multistep_epochs = None
        warmup_epochs = self.config['warmup_epochs']
        if mode == 'step':
            lr_step = self.config['lr_step']
        elif mode == 'multistep':
            multistep_epochs = self.config['multistep_epochs']
        return LR_Scheduler(self.config['lr_mode'], self.config['lr'], self.config['epochs'],
                            iters_per_epoch=len(self.train_loader), lr_step=lr_step, warmup_epochs=warmup_epochs,
                            multistep_epochs=multistep_epochs)

    def start(self):
        for i in range(0, self.config['epochs']):
            self.training(epoch=i)
            self.validation(epoch=i)
        self.logger.log('best val rmse: {}'.format(self.val_rmse))

    def find_lr(self):
        import math
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        num = len(self.train_loader) - 1
        final_value = 10.
        init_value = 1e-8
        lr = init_value
        mult = (final_value / init_value) ** (1 / num)
        beta = 0.98
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=0.00005, momentum=0.9)
        avg_loss = 0.
        best_loss = 0.
        batch_num = 0
        losses = []
        log_lrs = []

        tbar = tqdm(self.train_loader)
        for sample in tbar:
            batch_num += 1
            data = sample
            data.to(self.device, non_blocking=True)
            optimizer.zero_grad()
            preds = self.model(data)
            loss = self.criterion(preds, data.y)
            avg_loss = beta * avg_loss + (1-beta)*loss.item()
            smoothed_loss = avg_loss / (1-beta**batch_num)
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                break
            if smoothed_loss < best_loss or batch_num == 1:
                best_loss = smoothed_loss
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            loss.backward()
            optimizer.step()
            lr *= mult
            optimizer.param_groups[0]['lr'] = lr
        plt.plot(log_lrs[10:-5], losses[10:-5])
        plt.xlabel('learning rate(log scale)')
        plt.ylabel('loss')
        plt.savefig('find_lr_sgd.png')

    def __del__(self):
        self.writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml_100k', help='set dataset')
    args = parser.parse_args()
    assert args.dataset in ['ml_100k', 'ml_1m', 'flixster', 'yahoo_music', 'douban']
    if args.dataset == 'douban':
        args.dataset = 'Douban'
    trainer = Trainer(args.dataset)
    # trainer.find_lr()
    trainer.start()






