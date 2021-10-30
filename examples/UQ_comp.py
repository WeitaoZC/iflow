from operator import ne
import os, torch
from numpy.lib.function_base import average
import numpy as np
import scipy.io as spio
from scipy.linalg import expm, logm
import scipy.interpolate as ci
from iflow import model
from iflow.dataset import lasa_uq_dataset
from iflow.visualization import visualize_3d_generated_trj
import matplotlib.pyplot as plt

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'data')) + '/LASA_HandWriting_UQ/'


def find_pair(X, Y):
    if X.shape[0] > Y.shape[0]:
        N = X.shape[0] // Y.shape[0] + 1
    else:
        N = Y.shape[0] // X.shape[0] + 1
        # print(N)
    inds = []
    inds.append(0)
    for i in range(1, Y.shape[0]):
        X_inds = [j for j in range(inds[-1] + 1, min(N * (i + 1), X.shape[0] - (Y.shape[0] - i - 1)))]
        d = []
        for n in X_inds:
            d.append(np.linalg.norm(Y[i] - X[n]))
        ind = np.argmin(d)
        inds.append(ind + inds[-1] + 1)
    return inds


def R_Exp(R):
    uqs = np.zeros((R.shape[0],4))
    nR = np.linalg.norm(R,axis = 1)
    uqs[:,0] = np.cos(nR)
    uqs[:,1:] = R * np.sin(nR).reshape(R.shape[0],1)/nR.reshape(R.shape[0],1)
    return uqs

def quat_conj(q):
    q[1:] *= -1
    return q

def quat_prod(q1,q2):
    p = np.zeros(4)
    p[0] = q1[0] * q2[0] - q1[1:] @ q2[1:]
    p[1:] = q1[0] * q2[1:] + q2[0] * q1[1:] + np.cross(q1[1:],q2[1:])
    return p

def quat_Log(q):
    nEpsilon = np.linalg.norm(q[1:])
    qlog = np.arccos(q[0])*q[1:]/nEpsilon
    return qlog

def quat_log_err(q1,q2):
    err = 2*quat_Log(quat_prod(q1, quat_conj(q2)))
    err = np.linalg.norm(err)
    return err


def main_layer(dim, acti_func):
    return model.CouplingLayer(dim, nonlinearity=acti_func)


def create_flow_seq(dim, depth, acti_func):
    chain = []
    for i in range(depth):
        chain.append(main_layer(dim, acti_func))
        chain.append(model.RandomPermutation(dim))  # permutation for dimensions
        chain.append(model.LULinear(dim))  # LDU decomposition lower * diog * upper
    chain.append(main_layer(dim, acti_func))
    return model.SequentialFlow(chain)


# test
if __name__ == '__main__':
    filename = 'Sine_UQ'
    layers = 11
    activation_function = "ReLu"
    data = spio.loadmat(directory + filename + '.mat', squeeze_me=True)
    UQs = []
    for demo_i in data['demoUQ']:
        UQs.append(demo_i.tolist()[3].transpose()) #4*1000*4

    UQs = np.array(UQs)

    data = lasa_uq_dataset.LASA_UQ(filename=filename, device=device)
    dim = data.dim
    dynamics = model.TanhStochasticDynamics(dim, device=device, dt=0.003, T_to_stable=3)
    flow = create_flow_seq(dim, layers, activation_function)
    iflow = model.ContinuousDynamicFlow(dynamics=dynamics, model=flow, dim=dim).to(device)
    iflow.load_state_dict(torch.load(os.getcwd() + "/best_models/" + "Sine_SPD_11" +"_best.pt"))
    with torch.no_grad():
        iflow.eval()
        predicted_trajs = []
        for trj in data.train_data:
            n_trj = trj.shape[0]
            y0 = trj[0, :]
            y0 = torch.from_numpy(y0[None, :]).float().to(device)
            yn = trj[-1, :]
            yn = torch.from_numpy(yn[None, :]).float().to(device)
            traj_pred = iflow.generate_trj(y0, yn)
            traj_pred = traj_pred.detach().cpu().numpy()
            predicted_trajs.append(traj_pred)
        visualize_3d_generated_trj(data.train_data, predicted_trajs, device, fig_number=2)
    unnormlized = []
    for tri in predicted_trajs:
        unnormlized.append(data.unormalize(tri))

    gener_uqs = []
    LD_e = []
    for i in range(4):
        gener_uqs.append(R_Exp(unnormlized[i]))
    comp_inds = []
    for i in range(4):
        comp_inds.append(find_pair(predicted_trajs[i], data.train_data[i]))
        for j in range(len(comp_inds[i])):
                LD_e.append(quat_log_err(gener_uqs[i][comp_inds[i][j]],UQs[i,j]))
    aver_e = sum(LD_e) / 4000


    plt.plot([i for i in range(1000)], LD_e[0:1000], "r")
    plt.plot([i for i in range(1000)], LD_e[1000:2000], "g")
    plt.plot([i for i in range(1000)], LD_e[2000:3000], "k")
    plt.plot([i for i in range(1000)], LD_e[3000:4000], "b")
    plt.show()
    print("max error:{}".format(max(LD_e)))
    print("min error:{}".format(min(LD_e)))
    print("average error for {} :{}".format(filename, aver_e))

    # print(unnormlized[0][0])
    # print(gener_uqs[0][0])
    # print(UQs[0,0,:])

    # print(gener_uqs[0][comp_inds[0][10],:])
    # print(UQs[0,10,:])

