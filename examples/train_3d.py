import os, sys, time
import numpy as np
import torch
import torch.optim as optim
from iflow.dataset import lasa_3d_dataset
from torch.utils.data import DataLoader
from iflow import model
from iflow.trainers import goto_dynamics_train
from iflow.utils import to_numpy, to_torch
from iflow.visualization import visualize_latent_distribution, visualize_3dvector_field, visualize_3d_generated_trj
from iflow.test_measures import log_likelihood, iros_evaluation


percentage = .99
batch_size = 100
depth = 10
## optimization ##
lr = 0.001
weight_decay = 0.
## training variables ##
nr_epochs = 200
## filename ##
filename = 'NShape_SPD' #choose input data

######### GPU/ CPU #############
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

#### Invertible Flow model #####
def main_layer(dim):
    return  model.CouplingLayer(dim)


def create_flow_seq(dim, depth):
    chain = []
    for i in range(depth):
        chain.append(main_layer(dim))
        chain.append(model.RandomPermutation(dim))  #permutation for dimensions
        chain.append(model.LULinear(dim))   #LDU decomposition lower * diog * upper
    chain.append(main_layer(dim))
    return model.SequentialFlow(chain)


if __name__ == '__main__':
    ########## Data Loading #########
    data = lasa_3d_dataset.LASA3D(filename = filename, device = device)
    dim = data.dim
    params = {'batch_size': batch_size, 'shuffle': True}
    dataloader = DataLoader(data.dataset, **params)
    ######### Model #########
    dynamics = model.TanhStochasticDynamics(dim, device = device,dt=0.003, T_to_stable=3)
    #dynamics = model.LinearStochasticDynamics(dim, dt=0.01, T_to_stable=2.5)
    flow = create_flow_seq(dim, depth)
    iflow = model.ContinuousDynamicFlow(dynamics=dynamics, model=flow, dim=dim).to(device)
    ########## Optimization ################
    params = list(flow.parameters()) + list(dynamics.parameters())
    optimizer = optim.Adamax(params, lr = lr, weight_decay= weight_decay)
    #######################################
    
    for i in range(nr_epochs):
        cur_time = time.time()
        ## Training ##
        for local_x, local_y in dataloader:
            dataloader.dataset.set_step()
            optimizer.zero_grad()
            loss = goto_dynamics_train(iflow, local_x, local_y)#local_x:start point, local_y:（point after 20steps, 19, converge point）
            loss.backward(retain_graph=True)
            optimizer.step()
        
        print("1 epoch training time:{}".format(time.time() - cur_time))
        cur_time = time.time()

        ## Validation ##
        if i%20 == 0:
            with torch.no_grad():
                iflow.eval()
                print("-----epoch:{}/{}".format(i+1,nr_epochs))
                cur_time = time.time()
                predicted_trajs = []
                for trj in data.train_data:
                    n_trj = trj.shape[0]
                    y0 = trj[0, :]
                    y0 = torch.from_numpy(y0[None, :]).float().to(device)
                    yn = trj[-1, :]
                    yn = torch.from_numpy(yn[None, :]).float().to(device)
                    traj_pred = iflow.generate_trj( y0, yn)
                    traj_pred = traj_pred.detach().cpu().numpy()
                    predicted_trajs.append(traj_pred)
                print("generating time:{}".format(time.time() - cur_time))
                visualize_3d_generated_trj(data.train_data, predicted_trajs, device, fig_number=2)
                visualize_latent_distribution(data.train_data, iflow, device, fig_number=1)
                iros_evaluation(data.train_data, predicted_trajs, device)

                ## Prepare Data ##
                step = 20
                trj = data.train_data[0]
                trj_x0 = to_torch(trj[:-step,:], device)
                trj_x1 = to_torch(trj[step:,:], device)
                log_likelihood(trj_x0, trj_x1, step, iflow, device)
                print('The Variance of the latent dynamics are: {}'.format(torch.exp(iflow.dynamics.log_var)))
                print('The Velocity of the latent dynamics are: {}'.format(iflow.dynamics.Kv[0,0]))
                
                