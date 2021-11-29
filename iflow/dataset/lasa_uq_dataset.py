#coding=utf-8
"""
Self-defined data fromat to fit PyTorch dataloader function
"""
import os, sys, time
import numpy as np
import scipy.io as spio
import torch
from iflow.dataset.generic_dataset import Dataset

##### Dataset directory #####
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..','data')) + '/LASA_HandWriting_UQ/'

##### Self-defined dataset class #####
class LASA_UQ():
    def __init__(self, filename, device=torch.device('cpu')):
        self.filename = filename
        self.device = device
        useddata = spio.loadmat(directory + filename + '.mat', squeeze_me=True) # load data according to your file format
        self.trajs_real = []
        for demo_i in useddata['demoUQ']:
            self.trajs_real.append(demo_i.tolist()[0].transpose())
        trajs_np = np.asarray(self.trajs_real)
        self.n_trajs = trajs_np.shape[0]        #number of trajectories         
        self.trj_length = trajs_np.shape[1]     #length(points) of a trajectory 
        self.n_dims  = trajs_np.shape[2]        #points dimensions  
        self.dim = trajs_np.shape[2]            

        ##### Normalize trajectories #####
        trajs_np = np.reshape(trajs_np, (self.n_trajs * self.trj_length, self.n_dims))
        self.mean = np.mean(trajs_np,axis=0)
        self.std = np.std(trajs_np, axis=0)
        self.trajs_normalized = self.normalize(self.trajs_real)

        ##### Build Train Dataset #####
        self.train_data = []
        for i in range(self.trajs_normalized.shape[0]):
            self.train_data.append(self.trajs_normalized[i, ...])
        self.dataset = Dataset(trajs=self.train_data, device=device)

    def normalize(self, X):
        Xn = (X - self.mean)/self.std
        return Xn

    def unormalize(self, Xn):
        X = Xn*self.std + self.mean
        return X

##### test #####
if __name__ == "__main__":
    filename = 'Angle_UQ'
    device = torch.device('cpu')
    lasa3d = LASA_UQ(filename, device)
    print(lasa3d)