#coding=utf-8
'''
Script processing results from the well-trained model for uq and position data from real robot experiments
'''
import numpy as np
import torch
import torch.optim as optim
import os
from iflow.dataset import robot_uq_dataset
from torch.utils.data import DataLoader
from iflow import model
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from iflow.visualization import visualize_3d_generated_trj

def R_Exp(R):
    '''
    Exponential map: tangent space to manifold
    vectors (number * dimention) to unit quaternion (number * 4)
    '''
    uqs = np.zeros((R.shape[0],4))
    nR = np.linalg.norm(R,axis = 1)
    uqs[:,0] = np.cos(nR)
    uqs[:,1:] = R * np.sin(nR).reshape(R.shape[0],1)/nR.reshape(R.shape[0],1)
    return uqs

##### Model component #####
def main_layer(dim, acti_func):
    return  model.CouplingLayer(dim,nonlinearity = acti_func)

def create_flow_seq(dim, depth, acti_func):
    chain = []
    for i in range(depth):
        chain.append(main_layer(dim,acti_func))
        chain.append(model.RandomPermutation(dim))  #permutation for dimensions
        chain.append(model.LULinear(dim))   #LDU decomposition lower * diog * upper
    chain.append(main_layer(dim,acti_func))
    return model.SequentialFlow(chain)

if __name__ == "__main__":
    ##### Must keep the same parameter with the used model parameter file #####
    filename = "real_ori_vec"
    layers = 11
    activation_function = "ReLu"
    device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

    ##### load data for model and build the corresponding model for saved model file #####
    data = robot_uq_dataset.Robot_UQ(filename = filename, device = device)
    dim = data.dim
    dynamics = model.TanhStochasticDynamics(dim, device=device, dt=0.003, T_to_stable=3)
    flow = create_flow_seq(dim, layers, activation_function)
    iflow = model.ContinuousDynamicFlow(dynamics=dynamics, model=flow, dim=dim).to(device)
    iflow.load_state_dict(torch.load(os.getcwd() + "/robot_model/" + filename + "_11_" +"best.pt"))

    ##### generate new trajectories #####
    with torch.no_grad():
        iflow.eval()
        predicted_trajs = []
        ##### using the first point of the demonstrations as the starting point for new trajectories #####
        for trj in data.train_data:
            n_trj = trj.shape[0]
            y0 = trj[0, :]
            y0 = torch.from_numpy(y0[None, :]).float().to(device)
            yn = trj[-1, :]
            yn = torch.from_numpy(yn[None, :]).float().to(device)
            traj_pred = iflow.generate_trj(y0, yn)
            traj_pred = traj_pred.detach().cpu().numpy()
            predicted_trajs.append(traj_pred)
   
        ##### set new starting points #####
        new_start = np.zeros_like(data.train_data[0][0,:])
        for i in range(3):
            new_start += data.train_data[i][0, :]
        new_start /= 3
        # new_start[0] = -2
        # new_start[1] = -4
        # new_start[2] = 2
        # print(new_start)
        y0 = torch.from_numpy(new_start[None, :]).float().to(device)
        yn = data.train_data[0][-1, :]
        yn = torch.from_numpy(yn[None, :]).float().to(device)
        new_traj_pred = iflow.generate_trj(y0, yn)
        new_traj_pred = new_traj_pred.detach().cpu().numpy()
        # print(new_traj_pred.shape)
        # print("end point:{}".format(new_traj_pred[-1,:]))
        predicted_trajs.append(new_traj_pred)

        ##### unnormalized the generated and demonstration data on the tangent space #####
        unnormed_dem = data.unormalize(data.train_data)
        unnormed_pre = data.unormalize(predicted_trajs)

        ##### check the function for more details #####
        visualize_3d_generated_trj(unnormed_dem, unnormed_pre, device, fig_number=2, view1=40, view2=-80)
        # print(new_traj_pred.shape)

    ##### save generated uq data to txt #####
    # for i in range(len(unnormlized)):
    #     np.savetxt(os.getcwd()+ "/data/Robot_data/generated/ori_vec{}.txt".format(i+1), unnormlized[i])
    #     uqs = R_Exp(unnormlized[i])
    #     np.savetxt(os.getcwd()+ "/data/Robot_data/generated/ori_uq{}.txt".format(i+1), uqs)
