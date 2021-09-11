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
from iflow.trainers import goto_dynamics_train
from iflow.utils import to_numpy, to_torch
from iflow.visualization import visualize_latent_distribution, visualize_3dvector_field, visualize_3d_generated_trj
from iflow.test_measures import log_likelihood, iros_evaluation

percentage = .99
batch_size = 128
## optimization ##
weight_decay = 0.
## training variables ##
nr_epochs = 90
## filename ##
filename = 'GShape_SPD' #choose input data

######### GPU/ CPU #############
device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')


#set the random seed
SEED = 1

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


def objective(trial):

    ####set inner seed####
    seed = trial.number*2
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    ########## Data Loading #########
    data = lasa_spd_dataset.LASA_SPD(filename = filename, device = device)
    dim = data.dim
    params = {'batch_size': batch_size, 'shuffle': True}
    dataloader = DataLoader(data.dataset, **params)
    ######### Model #########
    dynamics = model.TanhStochasticDynamics(dim, device = device,dt=0.003, T_to_stable=3)
    #dynamics = model.LinearStochasticDynamics(dim, dt=0.01, T_to_stable=2.5)

    ###### set search range for the number of layers
    n_layers = trial.suggest_int("n_layers", 8, 12)
    # activation_func = trial.suggest_categorical("activation function", ["ReLu", "Tanh"])
    flow = create_flow_seq(dim, n_layers, "ReLu")
    iflow = model.ContinuousDynamicFlow(dynamics=dynamics, model=flow, dim=dim).to(device)
    ########## Optimization ################
    params = list(flow.parameters()) + list(dynamics.parameters())

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "Adamax"])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimizer = getattr(optim, optimizer_name)(params, lr=lr, weight_decay= 0)
    
    error = 10000
    for i in range(nr_epochs):
        ## Training ##
        for local_x, local_y in dataloader:
            dataloader.dataset.set_step()
            optimizer.zero_grad()
            loss = goto_dynamics_train(iflow, local_x, local_y)#local_x:start point, local_y:（point after 20steps, 19, converge point）
            loss.backward(retain_graph=True)
            optimizer.step()

        ## Validation ##
        if i%1 == 0:
            with torch.no_grad():
                iflow.eval()
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
                frechet_e, dtw_e = iros_evaluation(data.train_data, predicted_trajs, device)
                if dtw_e < error:
                    error = dtw_e
                    torch.save(iflow.state_dict(), os.getcwd() + "/search/saved_model/" + filename + "_trial" + str(trial.number) +"_best.pt")
        trial.report(error, nr_epochs)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    return error

if __name__ == '__main__':
    sampler = TPESampler(SEED)
    study = optuna.create_study(direction="minimize",sampler=sampler)
    study.optimize(objective, n_trials=80)

    joblib.dump(study, "study_1_8_relu_adamax.pkl")

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])    

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))     
                