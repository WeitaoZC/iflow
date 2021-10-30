import os, sys, time
import numpy as np
import torch
import torch.optim as optim
from iflow.dataset import lasa_spd_dataset
from torch.utils.data import DataLoader
from iflow import model
from iflow.trainers import goto_dynamics_train
from iflow.utils import to_numpy, to_torch
from iflow.visualization import visualize_latent_distribution, visualize_3d_generated_trj
from iflow.test_measures import log_likelihood, iros_evaluation
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("random training")


## hyperparameters 
depth = 11
activation_function = "ReLu"
# lr = 0.0009836349843763616
batch_size = 128
nr_epochs = 300

## other parameters
percentage = .99
weight_decay = 0.


######### GPU/ CPU #############
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

#set the random seed
# SEED = 48
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)

#### Invertible Flow model #####
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
    ## filename ##
    all_DTW = []
    for filename in os.listdir(os.getcwd()+ '/data/LASA_HandWriting_SPD/'):  #choose input data
    # ########## Data Loading #########
        (filename, extension) = os.path.splitext(filename)
    # filename = "Sine_SPD"
        data = lasa_spd_dataset.LASA_SPD(filename = filename, device = device)
        dim = data.dim
        params = {'batch_size': batch_size, 'shuffle': True}
        dataloader = DataLoader(data.dataset, **params)
        ######### Model #########
        dynamics = model.TanhStochasticDynamics(dim, device = device,dt=0.003, T_to_stable=3)
        #dynamics = model.LinearStochasticDynamics(dim, dt=0.01, T_to_stable=2.5)

        flow = create_flow_seq(dim, depth, activation_function)
        iflow = model.ContinuousDynamicFlow(dynamics=dynamics, model=flow, dim=dim).to(device)
        ########## Optimization ################
        params = list(flow.parameters()) + list(dynamics.parameters())
        optimizer = optim.Adam(params,weight_decay= 0)

        
        error = 10000
        DTW_loss = []
        for i in range(nr_epochs):
            # cur_time = time.time()
            ## Training ##
            for local_x, local_y in dataloader:
                dataloader.dataset.set_step()
                optimizer.zero_grad()
                loss = goto_dynamics_train(iflow, local_x, local_y)#local_x:start point, local_y:（point after 20steps, 19, converge point）
                loss.backward(retain_graph=True)
                optimizer.step()
            
            # print("1 epoch training time:{}".format(time.time() - cur_time))
            # cur_time = time.time()

            ## Validation ##
            if i%5 == 0:
                with torch.no_grad():
                    iflow.eval()
                    # print("-----epoch:{}/{}".format(i+1,nr_epochs))
                    # cur_time = time.time()
                    predicted_trajs = []
                    for trj in data.train_data:
                        # n_trj = trj.shape[0]
                        y0 = trj[0, :]
                        y0 = torch.from_numpy(y0[None, :]).float().to(device)
                        yn = trj[-1, :]
                        yn = torch.from_numpy(yn[None, :]).float().to(device)
                        traj_pred = iflow.generate_trj( y0, yn)
                        traj_pred = traj_pred.detach().cpu().numpy()
                        predicted_trajs.append(traj_pred)
                    # print("generating time:{}".format(time.time() - cur_time))
                    #visualize_3d_generated_trj(data.train_data, predicted_trajs, device, fig_number=2)
                    #visualize_latent_distribution(data.train_data, iflow, device, fig_number=1)
                    frechet_e, dtw_e = iros_evaluation(data.train_data, predicted_trajs, device)
                    # print('The DTW Distance is: {}'.format(dtw_e))
                    writer.add_scalar(filename+"11random", dtw_e, i)
                    DTW_loss.append(dtw_e)
                    if dtw_e < error:
                        error = dtw_e
                        torch.save(iflow.state_dict(), os.getcwd() + "/best_models/random/" + filename + "_11_random_" +"best.pt")

                    ## Prepare Data ##
                    # step = 20
                    # trj = data.train_data[0]
                    # trj_x0 = to_torch(trj[:-step,:], device)
                    # trj_x1 = to_torch(trj[step:,:], device)
                    # log_likeli = log_likelihood(trj_x0, trj_x1, step, iflow, device)
                    # writer.add_scalar('Log_likelihood', log_likeli, i)
                    #print('The Variance of the latent dynamics are: {}'.format(torch.exp(iflow.dynamics.log_var)))
                    #print('The Velocity of the latent dynamics are: {}'.format(iflow.dynamics.Kv[0,0]))
        all_DTW.append(DTW_loss)
    np.savetxt(os.getcwd() +"/best_models/random/all_dtw_e.npy", np.array(all_DTW))
    writer.close()