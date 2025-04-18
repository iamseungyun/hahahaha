import os
import logging
import argparse
import numpy as np
from utils.utils import *
import time

from exp.exp import Exp


def str2bool(v):
    return v.lower() in ('true')


def main(args):
    if (not os.path.exists(args.save_path)):
        mkdir(args.save_path)
        
    exp = Exp(args)

    if config.mode == 'train':
        exp.train()
        exp.test()
    elif config.mode == 'test':
        exp.test()

    return exp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--patch_size', type=str, default='357')
    parser.add_argument('--index', type=int, default=137)
    parser.add_argument('--dataset', type=str, default='SWAT')
    parser.add_argument('--anormly_ratio', type=float, default=1)
    parser.add_argument('--input_c', type=int, default=51)
    parser.add_argument('--output_c', type=int, default=51)

    # Model
    parser.add_argument('--model', type=str, default='DCdetector')
    parser.add_argument('--pretrain', type=bool, default=False)
    parser.add_argument('--pretrained_model_path', type=str, default='')

    # Mask
    parser.add_argument('--mask_save_path', type=str, default='')


    # Counterfactual

    
    # Device
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=True)
    parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')


    # Train
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--save_path', type=str, default='checkpoints')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)


    parser.add_argument('opts', help='Modify config options using the command-line', default=None, nargs=argparse.REMAINDER) 

    # Config
    parser.add_argument('--loss_fuc', type=str, default='MSE')
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--rec_timeseries', action='store_true', default=True)    


    
    
    config = parser.parse_args()
    args = vars(config)
    
    config.patch_size = [int(patch) for patch in config.patch_size.split('_')]

    if config.dataset == 'UCR':
        batch_size_buffer = [2,4,8,16,32,64,128,256]
        data_len = np.load('datasets/'+config.datasets + "/UCR_"+str(config.index)+"_train.npy").shape[0] 
        config.batch_size = find_nearest(batch_size_buffer, data_len / config.win_size)
    elif config.dataset == 'UCR_AUG':
        batch_size_buffer = [2,4,8,16,32,64,128,256]
        data_len = np.load('datasets/'+config.datasets + "/UCR_AUG_"+str(config.index)+"_train.npy").shape[0] 
        config.batch_size = find_nearest(batch_size_buffer, data_len / config.win_size)
    elif config.dataset == 'SMD_Ori':
        batch_size_buffer = [2,4,8,16,32,64,128,256,512]
        data_len = np.load('datasets/'+config.datasets + "/SMD_Ori_"+str(config.index)+"_train.npy").shape[0] 
        config.batch_size = find_nearest(batch_size_buffer, data_len / config.win_size)

    if config.mode == 'train':
        print("\n\n")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('================ Hyperparameters ===============')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('====================  Train  ===================')
    else:
        print('')
        print('what exp?')
        print('====================  Test  ===================')
        
    main(config)