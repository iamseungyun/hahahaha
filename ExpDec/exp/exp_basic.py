import os
import torch
from models.DCdetector import DCdetector as DCdetector, solver as DCdetector_solver

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'DCdetector': DCdetector, DCdetector_solver
        }
        self.solver_dict = {
            'DCdetector': DCdetector_solver
        }
        self.device = self._acquire_device()
        self.model, self.model_solver = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

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

    def mask_vali(self):
        pass


    def mask_train(self):
        pass


    def mask_test(self):
        pass
        

    def cf_model_train(self):
        pass

    
    def cf_model_test(self):
        pass
