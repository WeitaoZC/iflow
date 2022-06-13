#coding=utf-8
'''
Training script for Unit Quarernion(UQ) data in "data/LASA_HandWriting_UQ" or realexperiment "data/Robot_data/training_dataset"
'''
import os, sys, time
import numpy as np
import torch
import argparse
import torch.optim as optim
from iflow.dataset import robot_uq_dataset, lasa_uq_dataset
from torch.utils.data import DataLoader
from iflow import model
from iflow.trainers import goto_dynamics_train
from iflow.utils import to_numpy, to_torch
from iflow.visualization import visualize_latent_distribution, visualize_3d_generated_trj
from iflow.test_measures import log_likelihood, iros_evaluation
from torch.utils.tensorboard import SummaryWriter

##### directory to save training DTW loss in TensorBoard #####
writer = SummaryWriter("comp_training_UQ")

##### choose sub-dataset for training #####
# filename = 'real_ori_vec_pos'

##### Defalt Hyperparameters #####
depth = 11
activation_function = "ReLu"
lr = 0.0009836349843763616   #0.0013063372660080718
batch_size = 128
nr_epochs = 100

##### GPU/ CPU #####
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

##### Invertible Flow model #####
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


if __name__ == '__main__':
    ##### Data Loading #####
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--seed", type=int, default=48)
    parser.add_argument("--depth", type=int, default=11)
    parser.add_argument("--lr", type=float, default=0.0009836349843763616)
    args = parser.parse_args()
    ## hyperparameters 
    depth = args.depth

    ##### set the random seed #####
    SEED = 48
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    ##### training for all sub-dataset in dir #####
    all_DTW = dict()
    for filename in ['Multi_Models_2_UQ', 'Multi_Models_4_UQ', 'Saeghe_UQ', 'Sharpc_UQ', 'Sine_UQ', 'Spoon_UQ', 'Sshape_UQ', 'Trapezoid_UQ', 'WShape_UQ']:
	#os.listdir(os.getcwd()+ '/data/LASA_HandWriting_UQ/'):
        # (filename, extension) = os.path.splitext(filename)
        # if extension != '.mat':
            #continue
    
    # data = robot_uq_dataset.Robot_UQ(filename = filename, device = device) #this function should be defined by users refering "iflow/dataset/robot_uq_dataset.py"
        data = lasa_uq_dataset.LASA_UQ(filename = filename, device = device)
        dim = data.dim
        params = {'batch_size': batch_size, 'shuffle': True}
        dataloader = DataLoader(data.dataset, **params)

        ##### Model #####
        dynamics = model.TanhStochasticDynamics(dim, device = device,dt=0.001, T_to_stable=4)
        #dynamics = model.LinearStochasticDynamics(dim, dt=0.01, T_to_stable=2.5)
        flow = create_flow_seq(dim, depth, activation_function)
        iflow = model.ContinuousDynamicFlow(dynamics=dynamics, model=flow, dim=dim).to(device)

        ##### Optimizer #####
        params = list(flow.parameters()) + list(dynamics.parameters())
        optimizer = optim.Adam(params,lr = lr,weight_decay= 0) # choose the suitable optimizer for your problem

        ##### Training #####
        error = 10000 # manually set a huge error
        DTW_loss = []
        for i in range(nr_epochs):
            # cur_time = time.time()
            for local_x, local_y in dataloader:
                dataloader.dataset.set_step()
                optimizer.zero_grad()
                loss = goto_dynamics_train(iflow, local_x, local_y) #local_x:start point, local_y:（point after 20steps, interval step, converge point, index)
                loss.backward(retain_graph=True)
                optimizer.step()
            
            # print("1 epoch training time:{}".format(time.time() - cur_time))
            # cur_time = time.time()

            ##### Validation #####
            if i%10 == 0 or i>20:
                with torch.no_grad():
                    iflow.eval()
                    print("-----epoch:{}/{}".format(i+1,nr_epochs))
                    # cur_time = time.time()
                    predicted_trajs = []
                    for trj in data.train_data:
                        y0 = trj[0, :]
                        y0 = torch.from_numpy(y0[None, :]).float().to(device)
                        yn = trj[-1, :]
                        yn = torch.from_numpy(yn[None, :]).float().to(device)
                        traj_pred = iflow.generate_trj(y0, yn) # check the function to set the stop conditions
                        traj_pred = traj_pred.detach().cpu().numpy()
                        predicted_trajs.append(traj_pred)
                    # print("generating time:{}".format(time.time() - cur_time))

                    ##### user can visualize the generated trajectoried and demonstrations on the tangent space at each validation #####
                    #visualize_3d_generated_trj(data.train_data, predicted_trajs, device, fig_number=2)
                    #visualize_latent_distribution(data.train_data, iflow, device, fig_number=1)

                    ##### evaluate factors #####
                    frechet_e, dtw_e = iros_evaluation(data.train_data, predicted_trajs, device)
                    print('The DTW Distance is: {}'.format(dtw_e))

                    ##### Title for the loss curve in the TensorBoard #####
                    writer.add_scalar(filename + str(args.seed) + str(args.depth) + str(round(args.lr, 6)), dtw_e, i)
                    DTW_loss.append(dtw_e) # also saved in a list

                    ##### save the current model if its DTW loss is lower #####
                    if dtw_e < error:
                        error = dtw_e
                        # torch.save(iflow.state_dict(), os.getcwd() + "/robot_model/" + filename + "_11_" +"best.pt")
                        torch.save(iflow.state_dict(), os.getcwd() + "/best_models/direct_UQ/" + filename + str(args.depth) +"best.pt")
                    ##### Inside Information #####
                    # step = 20
                    # trj = data.train_data[0]
                    # trj_x0 = to_torch(trj[:-step,:], device)
                    # trj_x1 = to_torch(trj[step:,:], device)
                    # log_likeli = log_likelihood(trj_x0, trj_x1, step, iflow, device)
                    # writer.add_scalar('Log_likelihood', log_likeli, i)
                    #print('The Variance of the latent dynamics are: {}'.format(torch.exp(iflow.dynamics.log_var)))
                    #print('The Velocity of the latent dynamics are: {}'.format(iflow.dynamics.Kv[0,0]))
    ##### Save all loss changing if needed #####
        all_DTW[filename] = np.array(DTW_loss)
    np.save(os.getcwd() +"/best_models/direct_UQ/"+ str(args.depth) +"_all_dtw_e1.npy", all_DTW)
    writer.close()
