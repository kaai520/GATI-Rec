import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '1'

import torch
torch.multiprocessing.set_sharing_strategy('file_system')
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
from utils.logger import load_yaml
from utils.metrics import Evaluator
from dataloaders.datasets.MovieLens import DynamicMovieLens
from dataloaders.datasets.Flixster import Flixster
from dataloaders.datasets.Douban import Douban
from dataloaders.datasets.YahooMusic import YahooMusic


class Transfer(object):
    def __init__(self):
        self.config = load_yaml('config/transfer.yaml')
        self.device = 'cpu'
        self.cuda = self._init_gpu()  # if cuda, device:'cpu'->'cuda'
        self.model = self._init_model()
        if self.config['ensemble']:
            self.models = self._init_models()
        self.evaluator = Evaluator()

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
        attention = True
        model = GATIRec(input_channels=input_channels, EGAT_heads=heads, EGAT_output_channels=32, EGAT_layers=layers,
                        edge_classes=edge_classes, multiply_by=1, activation='elu', decoder_choice='mlp',
                        concat_nodes_feature=True, edge_embedding='cumsum', add_self_feature=True,
                        input_embedding=input_embedding, attention=attention)
        checkpoint = torch.load(self.config['checkpoint_path'])
        model.load_state_dict(checkpoint['state_dict'])
        if self.cuda:
            device_ids = [i for i in range(len(self.config['gpu_ids']))]
            if len(device_ids) > 1:
                model = DataParallel(model, device_ids=device_ids)
        model = model.to(self.device)
        return model

    def _init_models(self):
        input_embedding = self.config['input_embedding']
        edge_classes = self.config['edge_classes']
        heads = self.config['heads']
        layers = self.config['layers']
        input_channels = self.config['input_channels'] if input_embedding else 4
        attention = True
        checkpoint_epochs = [50, 60, 70, 80]
        models = []
        for epoch in checkpoint_epochs:
            model = GATIRec(input_channels=input_channels, EGAT_heads=heads, EGAT_output_channels=32, EGAT_layers=layers,
                            edge_classes=edge_classes, multiply_by=1, activation='elu', decoder_choice='mlp',
                            concat_nodes_feature=True, edge_embedding='cumsum', add_self_feature=True,
                            input_embedding=input_embedding, attention=attention)
            checkpoint_path = os.path.join(self.config['ensemble_path'],'checkpoint_epoch0{}.pth.tar'.format(epoch))
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['state_dict'])
            models.append(model)
        models = torch.nn.ModuleList(models)
        return models.to(self.device)



    def testing(self, dataset_name='ml_100k'):
        one_hot_flag = not self.config['input_embedding']
        if dataset_name == 'flixster':
            dataset = Flixster(root=self.config['dataset_root'],
                               max_neighbors=self.config['max_neighbors'], split='test',
                               one_hot_flag=one_hot_flag, transfer=True)
            self.model.set_multiply_by(self.config['flixster_multiply_by'])
        elif dataset_name == 'douban':
            dataset = Douban(root=self.config['dataset_root'], max_neighbors=self.config['max_neighbors'],
                             split='test', one_hot_flag=one_hot_flag)
            self.model.set_multiply_by(self.config['douban_multiply_by'])
        elif dataset_name == 'yahoo_music':
            dataset = YahooMusic(root=self.config['dataset_root'], max_neighbors=self.config['max_neighbors'],
                                 split='test', one_hot_flag=one_hot_flag, transfer=True)
            self.model.set_multiply_by(self.config['yahoo_music_multiply_by'])
        else:
            dataset = DynamicMovieLens(root=self.config['dataset_root'], dataset=self.config['pretrain_dataset'],
                                       max_neighbors=self.config['max_neighbors'],
                                       split='test', one_hot_flag=one_hot_flag)
            self.model.set_multiply_by(1)
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False,
                                num_workers=self.config['num_workers'], pin_memory=self.cuda)
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(dataloader)
        with torch.no_grad():
            for i, sample in enumerate(tbar):
                data = sample
                data.to(self.device, non_blocking=True)
                data_y = data.y
                preds = self.model(data)
                self.evaluator.add_batch(preds.cpu().numpy(), data_y.cpu().numpy())
            torch.cuda.empty_cache()
            rmse = self.evaluator.rmse()
            log_string = '[dataset: %s, numSamples: %d] rmse: %f' % (
            dataset_name, len(dataloader.dataset), rmse)
            print(log_string)

    def ensemble_testing(self, dataset_name='ml_100k'):
        one_hot_flag = not self.config['input_embedding']
        if dataset_name == 'flixster':
            dataset = Flixster(root=self.config['dataset_root'], max_neighbors=self.config['max_neighbors'],
                               split='test', one_hot_flag=one_hot_flag, transfer=True)
            for i in range(4):
                self.models[i].set_multiply_by(self.config['flixster_multiply_by'])
        elif dataset_name == 'douban':
            dataset = Douban(root=self.config['dataset_root'], max_neighbors=self.config['max_neighbors'], split='test',
                             one_hot_flag=one_hot_flag)
            for i in range(4):
                self.models[i].set_multiply_by(self.config['douban_multiply_by'])
        elif dataset_name == 'yahoo_music':
            dataset = YahooMusic(root=self.config['dataset_root'], max_neighbors=self.config['max_neighbors'],
                                 split='test', one_hot_flag=one_hot_flag, transfer=True)
            for i in range(4):
                self.models[i].set_multiply_by(self.config['yahoo_music_multiply_by'])
        else:
            dataset = DynamicMovieLens(root=self.config['dataset_root'], dataset=self.config['pretrain_dataset'],
                                       max_neighbors=self.config['max_neighbors'], split='test',
                                       one_hot_flag=one_hot_flag)
            for i in range(4):
                self.models[i].set_multiply_by(1)
        dataloader = DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False,
                                num_workers=self.config['num_workers'], pin_memory=self.cuda)
        self.models.eval()
        self.evaluator.reset()
        tbar = tqdm(dataloader)
        with torch.no_grad():
            for i, sample in enumerate(tbar):
                data = sample
                data.to(self.device, non_blocking=True)
                data_y = data.y
                outs = []
                for j in range(4):
                    preds = self.models[j](data).view(1, -1)
                    outs.append(preds)
                outs = torch.cat(outs, 0).mean(0)
                self.evaluator.add_batch(outs.cpu().numpy(), data_y.cpu().numpy())
            torch.cuda.empty_cache()
            rmse = self.evaluator.rmse()
            log_string = '[dataset: %s, numSamples: %d] rmse: %f' % (dataset_name, len(dataloader.dataset), rmse)
            print(log_string)

    def start(self):
        if self.config['ensemble']:
            self.ensemble_testing('flixster')
            self.ensemble_testing('douban')
            self.ensemble_testing('yahoo_music')
        else:
            self.testing('flixster')
            self.testing('douban')
            self.testing('yahoo_music')


if __name__ == '__main__':
    tr = Transfer()
    tr.start()

