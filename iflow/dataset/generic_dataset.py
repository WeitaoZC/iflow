import numpy as np
import os
import torch


class ContextualizedDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, trajs, contexts, device, steps=20):
        'Initialization'
        dim = trajs[0][0].shape[1]

        self.n_conditions = len(contexts)
        self.c_n = np.zeros((0, 3))
        self.x = []
        self.x_n = np.zeros((0, dim))
        self.x_n_ref = np.zeros((0, dim))

        for i in range(steps):
            self.c = np.zeros((0, 3))
            tr_i_all = np.zeros((0, dim))

            for i in range(len(trajs)):
                ref_trjs  = trajs[0]
                trj_list = trajs[i]
                context = contexts[i]
                for tr_i in trj_list:
                    _trj = tr_i[i:i - steps, :]
                    tr_i_all = np.concatenate((tr_i_all, _trj), 0)

                    c_all = np.concatenate((_trj.shape[0] * [context[None, :]]), 0)
                    self.c = np.concatenate((self.c, c_all), 0)

                    self.x_n = np.concatenate((self.x_n, tr_i[-1:, :]), 0)

                    tr_size = len(ref_trjs)
                    index_ref = np.random.randint(tr_size)
                    tr_ref = ref_trjs[index_ref]
                    self.x_n_ref = np.concatenate((self.x_n_ref, tr_ref[-1:, :]), 0)

                    self.c_n = np.concatenate((self.c_n, context[None, :]), 0)

            self.x.append(tr_i_all)


        self.x = torch.from_numpy(np.array(self.x)).float().to(device)
        self.x_n = torch.from_numpy(np.array(self.x_n)).float().to(device)
        self.x_n_ref = torch.from_numpy(np.array(self.x_n_ref)).float().to(device)

        self.c = torch.from_numpy(self.c).float().to(device)
        self.c_n = torch.from_numpy(self.c_n).float().to(device)

        self.len_n = self.x_n.shape[0]
        self.len = self.x.shape[1]
        self.steps_length = steps
        self.step = steps - 1

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def set_step(self, step=None):
        if step is None:
            self.step = np.random.randint(1, self.steps_length - 1)

    def __getitem__(self, index):
        'Generates one sample of data'

        X = self.x[0, index, :]
        X_1 = self.x[self.step, index, :]
        C = self.c[index, :]

        index = np.random.randint(self.len_n)
        X_N = self.x_n[index, :]
        C_N = self.c_n[index, :]
        X_REF_N = self.x_n_ref[index,:]

        return X, [X_1, int(self.step), X_N, C, C_N, X_REF_N]


class Dataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, trajs, device, steps=20):
        'Initialization'
        dim = trajs[0].shape[1] #(x,y)

        self.x = []
        self.x_n = np.zeros((0, dim))   #(0,2)
        for i in range(steps):
            tr_i_all = np.zeros((0,dim))
            for tr_i in  trajs:             #tr_i shape:(250,2)
                _trj = tr_i[i:i-steps,:]    #0:-20->0:230
                tr_i_all = np.concatenate((tr_i_all, _trj), 0)          #trajs[0][0:230]    shpae:(230*3,2)
                self.x_n = np.concatenate((self.x_n, tr_i[-1:,:]),0)    #trajs[0][-1]       shape:(3*20,2)
            self.x.append(tr_i_all) #(20,230*3,2)

        self.x = torch.from_numpy(np.array(self.x)).float().to(device)      #(20,230*3,2)
        self.x_n = torch.from_numpy(np.array(self.x_n)).float().to(device)  #(3*20,2)收敛位置，对同一个轨迹就是同一个点

        self.len_n = self.x_n.shape[0]  #60
        self.len = self.x.shape[1]      #230*3
        self.steps_length = steps       #20
        self.step = steps - 1           #19

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def set_step(self, step=None):
        if step is None:
            self.step = np.random.randint(1, self.steps_length-1)

    def __getitem__(self, index):
        'Generates one sample of data'

        X = self.x[0, index, :]
        X_1 = self.x[self.step, index, :]

        index = np.random.randint(self.len_n)
        X_N = self.x_n[index, :]

        return X, [X_1, int(self.step), X_N, index]
        #起始位置  20steps后位置   19  随机一个收敛点(都是一样的) 随机数


class CycleDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, trajs, device, trajs_phase, steps=20):
        'Initialization'
        dim = trajs[0].shape[1]

        self.x = []
        self.x_n = np.zeros((0, dim))
        for i in range(steps):
            tr_i_all = np.zeros((0,dim))
            for tr_i in  trajs:
                _trj = tr_i[i:i-steps,:]
                tr_i_all = np.concatenate((tr_i_all, _trj), 0)
                self.x_n = np.concatenate((self.x_n, tr_i[-1:,:]),0)
            self.x.append(tr_i_all)

        self.x = torch.from_numpy(np.array(self.x)).float().to(device)
        self.x_n = torch.from_numpy(np.array(self.x_n)).float().to(device)

        ## Phase ordering ##
        trp_all = np.zeros((0))
        for trp_i in  trajs_phase:
            _trjp = trp_i[:-steps]
            trp_all = np.concatenate((trp_all, _trjp), 0)
        self.trp_all = torch.from_numpy(trp_all).float().to(device)

        self.len_n = self.x_n.shape[0]
        self.len = self.x.shape[1]
        self.steps_length = steps
        self.step = steps - 1

    def __len__(self):
        'Denotes the total number of samples'
        return self.len

    def set_step(self, step=None):
        if step is None:
            self.step = np.random.randint(1, self.steps_length-1)

    def __getitem__(self, index):
        'Generates one sample of data'

        X = self.x[0, index, :]
        X_1 = self.x[self.step, index, :]
        phase = self.trp_all[index]

        return X, [X_1, int(self.step), phase]
