import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
from utils.utils import *
from model.DCdetector import DCdetector
from data_factory.data_loader import get_loader_segment
from einops import rearrange
from metrics.metrics import *
from tqdm import tqdm
import csv
import warnings
warnings.filterwarnings('ignore')
import json
import pandas as pd
from scipy.ndimage import uniform_filter1d
from tabulate import tabulate


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)

def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
#250306: adaptive threshold
def adaptive_threshold(opt, combined_energy, window_size=10, percentile=99.5, k=1.5):
    print('adaptive threshold: ', opt)
    if opt=='quantile':
        series = pd.Series(combined_energy)
        adaptive_thresh = series.rolling(window=window_size, min_periods=1).quantile(percentile / 100).to_numpy()
    elif opt=='ma':
        moving_avg = uniform_filter1d(combined_energy, size=window_size, mode='nearest')
        moving_std = np.sqrt(uniform_filter1d((combined_energy - moving_avg) ** 2, size=window_size, mode='nearest'))
        adaptive_thresh = moving_avg + k * moving_std
    else:
        print('opt unnamed.')
    return adaptive_thresh

#250311: additional loss for dual attention
#series, prior: length 6 list (each element 256(batch) x 1(head) x 105(time-length) x 105(time-length))
def compute_al(series, prior, k=1.0, alpha=0.5, yet=False):
    if not yet:
        contrastive_loss, variance_loss = 0., 0.
    else:
        series = torch.stack(series, dim=0)
        #prior = torch.stack(prior, dim=0)
        
        series_variance = torch.var(series, dim=[-2, -1], keepdim=True)
        #prior_variance = torch.var(prior, dim=[-2, -1], keepdim=True)
        variance_loss = torch.mean(series_variance) 
        
        # 1) Contrastive Loss (Patch-wise Attention)
#         patch_similarity = torch.matmul(series, prior.transpose(-2, -1)) 
#         pos_pairs = torch.sum(patch_similarity, dim=-1)
#         neg_pairs = torch.sum(patch_similarity, dim=-1)

#         # Contrastive Loss (maximize similarity within anomaly and minimize it across different segments)
#         contrastive_loss = -torch.mean(torch.log(torch.exp(pos_pairs) / torch.exp(neg_pairs)))
        contrastive_loss = 0.

    return [contrastive_loss, variance_loss]

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss

        
class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        self.train_loader = get_loader_segment(self.index, self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='train', dataset=self.dataset, )
        self.vali_loader = get_loader_segment(self.index, self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, self.data_path, batch_size=self.batch_size, win_size=self.win_size, mode='thre', dataset=self.dataset)

        self.device = torch.device(f"cuda:{self.gpu}" if torch.cuda.is_available() else "cpu")
        self.build_model()
       
        if self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
        

    def build_model(self):
        self.model = DCdetector(win_size=self.win_size, enc_in=self.input_c, c_out=self.output_c, n_heads=self.n_heads, d_model=self.d_model, e_layers=self.e_layers, patch_size=self.patch_size, channel=self.input_c)
        
        if torch.cuda.is_available():
            self.model.to(self.device)
            
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        
    def vali(self, vali_loader):
        self.model.eval()
        val_loss = 0
        loss = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               self.win_size)).detach())) + torch.mean(
                    my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)).detach(),
                        series[u])))
                prior_loss += (torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)),
                               series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(),
                               (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       self.win_size)))))
                
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)
            
            
            kl_loss = prior_loss - series_loss 
                    
            #250311 todo: additional loss for Patch-wise Attention & In-Patch Attention                    
            lambda_params = [1.0, 1.0]  
            als = compute_al(series, prior, yet=False)  
            total_loss = kl_loss + sum(h * ls for h, ls in zip(lambda_params, als))

            val_loss += total_loss.item()

            loss.append(val_loss)

        return np.average(loss)


    def train(self):

        time_now = time.time()
        path = self.model_save_path
        if not os.path.exists(path):
            os.makedirs(path)
        early_stopping = EarlyStopping(patience=5, verbose=True, dataset_name=self.data_path)
        train_steps = len(self.train_loader)

        for epoch in range(self.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            train_loss = 0
            
            with tqdm(total=train_steps, desc=f"Epoch {epoch+1}/{self.num_epochs}", unit="batch") as pbar:
                for i, (input_data, labels) in enumerate(self.train_loader):
                    self.optimizer.zero_grad()
                    input = input_data.float().to(self.device)
                    series, prior = self.model(input)  #6 * [256, 1, 105, 105] 

                    series_loss = 0.0
                    prior_loss = 0.0

                    for u in range(len(prior)):
                        # Normalize prior
                        prior_sum = torch.sum(prior[u], dim=-1, keepdim=True)
                        normalized_prior = prior[u] / prior_sum.repeat(1, 1, 1, self.win_size)

                        # Compute KL losses
                        series_kl_loss = torch.mean(my_kl_loss(series[u], normalized_prior.detach())) + \
                                         torch.mean(my_kl_loss(normalized_prior.detach(), series[u]))

                        prior_kl_loss = torch.mean(my_kl_loss(normalized_prior, series[u].detach())) + \
                                        torch.mean(my_kl_loss(series[u].detach(), normalized_prior))
                        
                        series_loss += series_kl_loss
                        prior_loss += prior_kl_loss

                    series_loss = series_loss / len(prior)
                    prior_loss = prior_loss / len(prior)
                    
                    #print('train_lp: ', prior_loss.item(), 'train_ln: ',  series_loss.item())

                    kl_loss = prior_loss - series_loss 
                    
                    #250311 todo: additional loss for Patch-wise Attention & In-Patch Attention                    
                    lambda_params = [1.0, 1.0]  
                    als = compute_al(series, prior, yet=False)  
                    loss = kl_loss + sum(h * ls for h, ls in zip(lambda_params, als))
                    
                    train_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()

                    pbar.set_postfix(loss=loss.item())  # Update tqdm bar with loss
                    pbar.update(1)

            
#             for i, (input_data, labels) in enumerate(self.train_loader):

#                 self.optimizer.zero_grad()
#                 iter_count += 1
#                 input = input_data.float().to(self.device)
#                 series, prior = self.model(input)
                
#                 series_loss = 0.0
#                 prior_loss = 0.0

#                 for u in range(len(prior)):
#                     series_loss += (torch.mean(my_kl_loss(series[u], (
#                             prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                    self.win_size)).detach())) + torch.mean(
#                         my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                            self.win_size)).detach(),
#                                    series[u])))
#                     prior_loss += (torch.mean(my_kl_loss(
#                         (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                 self.win_size)),
#                         series[u].detach())) + torch.mean(
#                         my_kl_loss(series[u].detach(), (
#                                 prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
#                                                                                                        self.win_size)))))

#                 series_loss = series_loss / len(prior)
#                 prior_loss = prior_loss / len(prior)

#                 loss = prior_loss - series_loss 
#                 train_loss += loss.item() 

#                 if (i + 1) % 100 == 0:
#                     speed = (time.time() - time_now) / iter_count
#                     left_time = speed * ((self.num_epochs - epoch) * train_steps - i)
#                     print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
#                     iter_count = 0
#                     time_now = time.time()
 
#                 loss.backward()
#                 self.optimizer.step()

           
            train_loss /= train_steps
            vali_loss = self.vali(self.test_loader)

            print(
                "Epoch: {0}, Cost time: {1:.3f}s, Train Loss: {2:.6f}, Validation Loss: {3:.6f}".format(
                    epoch + 1, time.time() - epoch_time, train_loss, vali_loss))
    
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

            
    def test(self):
        self.model.load_state_dict(
            torch.load(
                os.path.join(str(self.model_save_path), str(self.data_path) + '_checkpoint.pth')))
        self.model.eval()
        temperature = 50

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input) #each: 6 x (256, 1, 105, 105)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
            
            loss = prior_loss - series_loss
            #metric = torch.softmax(loss, dim=-1)
            #metric = torch.softmax(loss / 100, dim=-1)
            #metric = torch.sigmoid(loss)
            metric = torch.softmax((-series_loss - prior_loss), dim=-1) #original code - 두 loss 중 뭐가 큰지 모르니까? 
            cri = metric.detach().cpu().numpy() #256, 105
            attens_energy.append(cri)
            
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy) #74350080, 

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature

            loss = prior_loss - series_loss
            #metric = torch.softmax(loss, dim=-1)
            #metric = torch.softmax(loss / 100, dim=-1)
            #metric = torch.sigmoid(loss)
            metric = torch.softmax((-series_loss - prior_loss), dim=-1) #original code
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.anormly_ratio) #top 0.5%
        #thresh = adaptive_threshold(opt='quantile', combined_energy=test_energy, percentile=100 - self.anormly_ratio)
        
        #np.save('wj_temp/threshold_quantile.npy', thresh)
        print("Threshold :", thresh)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input) #b, h, l, l

            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss = my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                else:
                    series_loss += my_kl_loss(series[u], (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   self.win_size)).detach()) * temperature
                    prior_loss += my_kl_loss(
                        (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                self.win_size)),
                        series[u].detach()) * temperature
                    
            
            loss = prior_loss - series_loss
            #metric = torch.softmax(loss, dim=-1)
            #metric = torch.softmax(loss / 100, dim=-1)
            #metric = torch.sigmoid(loss)
            metric = torch.softmax((-series_loss - prior_loss), dim=-1) #original code
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)
            
        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        pred = (test_energy > thresh).astype(int)
        gt = test_labels.astype(int)
        
        #np.save("wj_temp/pred_new_qt.npy", pred)
        #np.save("wj_temp/test_energy_new_qt.npy", test_energy)
        #np.save("wj_temp/gt.npy", gt)
        
        matrix = [self.index]
        scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)
        
        table_data = [[metric, f"{score:.4f}"] for metric, score in scores_simple.items()]
        print(tabulate(table_data, headers=["Metric", "Score"], tablefmt="grid"))
    
#         for key, value in sorted(scores_simple.items()):
#             print(len(scores_simple))
#             matrix.append(value)
#             print('{0:21} : {1:0.4f}'.format(key, value))

#         anomaly_state = False
#         for i in range(len(gt)):
#             if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
#                 anomaly_state = True
#                 for j in range(i, 0, -1):
#                     if gt[j] == 0:
#                         break
#                     else:
#                         if pred[j] == 0:
#                             pred[j] = 1
#                 for j in range(i, len(gt)):
#                     if gt[j] == 0:
#                         break
#                     else:
#                         if pred[j] == 0:
#                             pred[j] = 1
#             elif gt[i] == 0:
#                 anomaly_state = False
#             if anomaly_state:
#                 pred[i] = 1

#         pred = np.array(pred)
#         gt = np.array(gt)
        
        #np.save("wj_temp/pred_adj_new_qt.npy", pred)

#         from sklearn.metrics import precision_recall_fscore_support
#         from sklearn.metrics import accuracy_score

#         accuracy = accuracy_score(gt, pred)
#         precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
#         print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(accuracy, precision, recall, f_score))
        
        if self.data_path in ['UCR', 'UCR_AUG']:
            with open('result/'+self.data_path+'.csv', 'a+') as f:
                writer = csv.writer(f)
                writer.writerow(matrix)

#         return accuracy, precision, recall, f_score
