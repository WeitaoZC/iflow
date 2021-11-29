#coding=utf-8
'''
Script for Hyperparameter search
'''
import os, sys, time
import numpy as np
import torch
import optuna
from optuna.trial import TrialState
from optuna.samplers import TPESampler
import joblib
import torch.optim as optim
from iflow.dataset import lasa_spd_dataset
from torch.utils.data import DataLoader
from iflow import model
import argparse
from iflow.trainers import goto_dynamics_train
from iflow.utils import to_numpy, to_torch
from iflow.visualization import visualize_latent_distribution, visualize_3d_generated_trj
from iflow.test_measures import log_likelihood, iros_evaluation

######### GPU/ CPU #############
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')

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

##### Using super function from Optuna #####
def objective(trial):
    ##### set inner seed for each trial#####
    seed = trial.number # user can change as their wish
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    ##### Data Loading #####
    data = lasa_spd_dataset.LASA_SPD(filename = args.file, device = device)
    dim = data.dim
    params = {'batch_size': args.batchsize, 'shuffle': True}
    dataloader = DataLoader(data.dataset, **params)

    ##### set search range for the number of layers #####
    n_layers = trial.suggest_int("n_layers", 8, 12)

    ##### set options for activation funtion #####
    activation_func = trial.suggest_categorical("activation function", ["ReLu", "Tanh"])

    ##### set options for optimizer #####
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "Adamax"])

    ##### set the search range for learning rate #####
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

    ##### Model #####
    dynamics = model.TanhStochasticDynamics(dim, device = device,dt=0.003, T_to_stable=3)
    #dynamics = model.LinearStochasticDynamics(dim, dt=0.01, T_to_stable=2.5)
    flow = create_flow_seq(dim, n_layers, activation_func)
    iflow = model.ContinuousDynamicFlow(dynamics=dynamics, model=flow, dim=dim).to(device)

    ##### Optimization #####
    params = list(flow.parameters()) + list(dynamics.parameters())
    optimizer = getattr(optim, optimizer_name)(params, lr=lr, weight_decay= 0)
    
    ##### Training #####
    error = 10000 # manually set a huge error
    for i in range(args.epoch):
        for local_x, local_y in dataloader:
            dataloader.dataset.set_step()
            optimizer.zero_grad()
            loss = goto_dynamics_train(iflow, local_x, local_y)
            loss.backward(retain_graph=True)
            optimizer.step()

        if i%1 == 0:
            with torch.no_grad():
                iflow.eval()
                predicted_trajs = []
                for trj in data.train_data:
                    y0 = trj[0, :]
                    y0 = torch.from_numpy(y0[None, :]).float().to(device)
                    yn = trj[-1, :]
                    yn = torch.from_numpy(yn[None, :]).float().to(device)
                    traj_pred = iflow.generate_trj(y0, yn) # check the function to set the stop conditions
                    traj_pred = traj_pred.detach().cpu().numpy()
                    predicted_trajs.append(traj_pred)
                frechet_e, dtw_e = iros_evaluation(data.train_data, predicted_trajs, device)
                if dtw_e < error:
                    error = dtw_e
                    torch.save(iflow.state_dict(), os.getcwd() + "/search/saved_model/" + args.file + "_trial" + str(trial.number) +"_best.pt")
        trial.report(error, args.epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return error

if __name__ == '__main__':
    ##### change parameters as user sets
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--seed", type=int, default=1) #outer seed for Optuna
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--file", type=str, default="NShape_SPD")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batchsize", type=int, default=128)
    args = parser.parse_args()
    sampler = TPESampler(args.seed)
    study = optuna.create_study(direction="minimize",sampler=sampler)
    study.optimize(objective, n_trials=args.trials)

    ##### save searching information #####
    joblib.dump(study, os.getcwd() + "/search/study_"+str(args.seed)+ args.file+".pkl")

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])    

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial for {}:".format(args.file))
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))     
                