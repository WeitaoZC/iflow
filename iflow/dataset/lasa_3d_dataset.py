import os, sys, time
import numpy as np
import scipy.io as spio
import torch
from iflow.dataset.generic_dataset import Dataset


directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..','data')) + '/LASA_HandWriting_SPD/'


class LASA3D():
    def __init__(self, filename, device=torch.device('cpu')):

        ## Define Variables and Load trajectories ##
        self.filename = filename
        self.dim = 3
        self.device = device
        useddata = spio.loadmat(directory + filename + '.mat', squeeze_me=True)
        self.trajs_real = []
        for demo_i in useddata['demoSPD']:
            self.trajs_real.append(demo_i.tolist()[0].transpose())
        trajs_np = np.asarray(self.trajs_real)
        self.n_trajs = trajs_np.shape[0]        #number of trajectories         4
        self.trj_length = trajs_np.shape[1]     #length(points) of a trajectory 1000
        self.n_dims  = trajs_np.shape[2]        #points dimensions              3(x,y,z)

        ## Normalize trajectories ##
        trajs_np = np.reshape(trajs_np, (self.n_trajs * self.trj_length, self.n_dims))
        self.mean = np.mean(trajs_np,axis=0)
        self.std = np.std(trajs_np, axis=0)
        self.trajs_normalized = self.normalize(self.trajs_real)

        ## Build Train Dataset
        self.train_data = []
        for i in range(self.trajs_normalized.shape[0]):
            self.train_data.append(self.trajs_normalized[i, ...])   #(4*1000*3)
        self.dataset = Dataset(trajs=self.train_data, device=device)

    def normalize(self, X):
        Xn = (X - self.mean)/self.std
        return Xn

    def unormalize(self, Xn):
        X = Xn*self.std + self.mean
        return X


if __name__ == "__main__":
    filename = 'Angle_SPD'
    device = torch.device('cpu')
    lasa3d = LASA3D(filename, device)
    print(lasa3d)