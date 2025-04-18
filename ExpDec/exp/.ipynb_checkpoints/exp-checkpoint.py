import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from einops import rearrange
from tqdm import tqdm
import csv
from tabulate import tabulate

from data_factory.data_loader import get_loader_segment
from exp.exp_basic import Exp_Basic
from counterfactual.diffusion_models.gaussian_diffusion import Diffusion_TS
            

class Exp(Exp_Basic):
    def __init__(self, args):
        super(Exp, self).__init__(args)
        self.args = args

        self.train_loader = get_loader_segment(self.args.index, dataset=self.args.dataset, batch_size=self.args.batch_size, win_size=self.args.win_size, mode='train')
        self.vali_loader = get_loader_segment(self.args.index, dataset=self.args.dataset, batch_size=self.args.batch_size, win_size=self.args.win_size, mode='val')
        self.test_loader = get_loader_segment(self.args.index, dataset=self.args.dataset, batch_size=self.args.batch_size, win_size=self.args.win_size, mode='test')
        self.thre_loader = get_loader_segment(self.args.index, dataset=self.args.dataset, batch_size=self.args.batch_size, win_size=self.args.win_size, mode='thre')

        self.device = self._acquire_device()

        self.mask_generator = MaskGenerator(self.args.enc_in, self.args.seq_len) # input dim이 enc_in 맞나??
        self.model = self._build_model()
        self.diffusion = self._build_diffusion_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        # Anomaly dtection 모델 dict 
        path = f'./models/{self.args.model}/model_config/{self.args.dataset}.yaml'
        self.model_config = load_yaml_config(path)
        
        self.model = self.model_dict[self.args.model].Model(self.model_config.model).float().to(self.device) #TODO model_config 만들기
        self.solver = self.solver_dict[self.args.model].Model(self.model_config.solver).float().to(self.device)

        if self.args.pretrain:
            if self.args.pretrained_model_path = None:
                self.model.load_state_dict(torch.load(os.path.join(str(self.model_config.model_save_path), str(self.model_config.data_path) + '_checkpoint.pth')))
            else:
                self.model.load_state_dict(torch.load(self.args.pretrained_model_path))
        else:
            self.model = self.solver.train()

        for param in self.model.parameters():
            param.requires_grad = False
        return self.model

    def _build_diffusion_model(self):
        self.diffusion = Diffusion_TS(seq_length=self.args.win_size, feature_size=self.args.input_c, **kwargs) ## feature_dim 뭘까?
        return self.diffusion

    def train(self):
        self.mask_train()
        self.cf_model_train()

    def test(self):
        self.mask_test()
        self.cf_model_test()


    def mask_vali(self, vali_loader):
        self.mask_generator.eval()
        pass


    def mask_train(self):
        pass


    def mask_test(self):
        pass
        

    def cf_model_train(self):
        pass

    
    def cf_model_test(self):
        pass
        
            


        