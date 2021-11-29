#coding=utf-8
'''
Script processing results from the well-trained model for artificially made spd data
'''
import os, torch
from numpy.lib.function_base import average
import numpy as np
import scipy.io as spio
from scipy.linalg import expm, logm
import scipy.interpolate as ci
from iflow import model
from iflow.dataset import lasa_spd_dataset
from iflow.visualization import visualize_3d_generated_trj, visualize_vector_field
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'data')) + '/LASA_HandWriting_SPD/'


def find_pair(X, Y):
    '''
    X for generated trajectories from model
    Y for demonstration trajectories
    for each point in Y return the index of its nearest point in X
    '''
    if X.shape[0] > Y.shape[0]:
        N = X.shape[0] // Y.shape[0] + 1
    else:
        N = Y.shape[0] // X.shape[0] + 1
        # print(N)
    inds = []
    inds.append(0)
    for i in range(1, Y.shape[0]):
        X_inds = [j for j in range(inds[-1], min(N * (i + 1), X.shape[0] - (Y.shape[0] - i - 1)))]
        d = []
        for n in X_inds:
            d.append(np.linalg.norm(Y[i] - X[n]))
        ind = np.argmin(d)
        inds.append(ind + inds[-1])
    return inds


def vec2sym_mat(V):
    '''
    Transform list of vectors to tensor of symmetric matrices
    '''
    n_trj = len(V)
    n = [len(V[i]) for i in range(n_trj)]
    d = 3
    D = int((-1 + np.sqrt(1 + 8 * d)) // 2)
    M = []
    for i in range(n_trj):
        for j in range(n[i]):
            v = V[i][j]
            m = np.diag(v[:D])
            ind = np.cumsum(range(D, 0, -1))
            for k in range(D - 1):
                m = m + np.diag(v[ind[k]:], k + 1) / np.sqrt(2) + np.diag(v[ind[k]:], -k - 1) / np.sqrt(2)  # Mandel notation
            M.append(m)
    return M


def exp_map(U):
    '''
    exponential map: tangent space to manifold
    one symmetric matrix to one SPD w.r.t. [[100,0],[0,100]]
    '''
    P = np.array([[10, 0], [0, 10]])
    N = np.array([[0.1, 0], [0, 0.1]])
    m = P @ expm(N @ U @ N) @ P

    return m


def Log_Euclidean_d(A, B):
    '''
    distance bewteen SPD A and B
    '''
    res = np.linalg.norm(logm(B) - logm(A), ord='fro')
    return res

##### Model component #####
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


if __name__ == '__main__':
    ##### Must keep the same parameter with the used model file #####
    layers = 11
    activation_function = "ReLu"

    ##### to save comparison results #####
    # fo = open(os.getcwd() +"/results/error/error.txt","w")

    ##### give subdataset (for all subdataset in the dir or choose one) #####
    # for filename in os.listdir(os.getcwd()+ '/data/LASA_HandWriting_SPD/'):  #choose input data
    #     (filename, extension) = os.path.splitext(filename)
    filename = 'Multi_Models_3'

    ##### load original data to acquire demonstration data #####
    data = spio.loadmat(directory + filename + '_SPD.mat', squeeze_me=True)
    SPDs = []
    for demo_i in data['demoSPD']:
        SPDs.append(demo_i.tolist()[3])

    SPDs = np.array(SPDs)
    SPD_m = []
    for j in range(SPDs.shape[0]):
        for i in range(SPDs.shape[-1]):
            m = np.array([SPDs[j][0][0][i], SPDs[j][0][1][i], SPDs[j][1][0][i], SPDs[j][1][1][i]]).reshape(2, 2)
            SPD_m.append(m)

    ##### load data for model and build the corresponding model for saved model file #####
    data = lasa_spd_dataset.LASA_SPD(filename=filename+"_SPD", device=device)
    dim = data.dim
    dynamics = model.TanhStochasticDynamics(dim, device=device, dt=0.003, T_to_stable=3)
    flow = create_flow_seq(dim, layers, activation_function)
    iflow = model.ContinuousDynamicFlow(dynamics=dynamics, model=flow, dim=dim).to(device)
    iflow.load_state_dict(torch.load(os.getcwd() + "/best_models/11_best/"+ filename + "_SPD.pt"))

    ##### generate new trajectories #####
    with torch.no_grad():
        iflow.eval()
        predicted_trajs = []

        ##### using the first point of the demonstrations as the starting point for new trajectories 
        ##### can be changed
        for trj in data.train_data:
            y0 = trj[0, :]
            y0 = torch.from_numpy(y0[None, :]).float().to(device)
            yn = trj[-1, :]
            yn = torch.from_numpy(yn[None, :]).float().to(device)
            traj_pred = iflow.generate_trj(y0, yn)
            traj_pred = traj_pred.detach().cpu().numpy()
            predicted_trajs.append(traj_pred)
        ##### Visualize the genarated trajectories and demonstrations #####
        # visualize_3d_generated_trj(data.train_data, predicted_trajs, device, fig_name=filename)
        # visualize_vector_field(data.train_data, iflow, predicted_trajs, device, fig_name=filename)

    ##### unnormalized the generated data #####
    unnormlized = []
    for tri in predicted_trajs:
        unnormlized.append(data.unormalize(tri))

    ##### convert vector to symmetric matrix #####
    sym_m = vec2sym_mat(unnormlized)

    ##### find corresponding pairs from generated data and demonstration to compare #####
    comp_inds = []
    LD_e = []
    begin = 0
    for i in range(4):
        comp_inds.append(find_pair(predicted_trajs[i], data.train_data[i]))
        for j in range(1000):
            LD_e.append(Log_Euclidean_d(exp_map(sym_m[begin + comp_inds[i][j]]), SPD_m[i * 1000 + j]))
        begin += len(unnormlized[i])
    aver_e = sum(LD_e) / 4000

    ##### check computations #####
    # print("choice:{}".format(comp_inds[0][500]))
    # print("generated:{}".format(predicted_trajs[0][comp_inds[0][500],:]))
    # print("unnormed:{}".format(unnormlized[0][comp_inds[0][500],:]))
    # print("symmetric matrix:{}".format(sym_m[comp_inds[0][500]]))
    # print("generaed SPD:{}".format(exp_map(sym_m[comp_inds[0][500]])))
    # print("given SPD:{}".format(SPD_m[500]))

    ##### save generated SPD(2*2*1000) to .mat #####
    generated_SPD = []
    begin = 0
    for i in range(len(predicted_trajs)):
        SPD1 = np.zeros((2,2,1000))
        for j in range(1000):
            SPD1[:, :, j] = exp_map(sym_m[begin+ comp_inds[i][j]])
        generated_SPD.append(SPD1)
        begin += len(unnormlized[i])
    spio.savemat("D:/Pytorch/iflow/iflow/data/generated_spd/"+filename+".mat", {"SPD1":generated_SPD[0], 
        "SPD2":generated_SPD[1],"SPD3":generated_SPD[2],"SPD4":generated_SPD[3]})

    ##### plot the comparison results (distance of SPD pairs)#####
    plt.figure(figsize=(10, 5))
    plt.plot([i for i in range(1000)], LD_e[0:1000], "k")
    plt.plot([i for i in range(1000)], LD_e[1000:2000], "g")
    plt.plot([i for i in range(1000)], LD_e[2000:3000], "r")
    plt.plot([i for i in range(1000)], LD_e[3000:4000], "b")
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.ylabel("Log Euclidean distanc")
    # plt.savefig(os.getcwd() +"/results/error/"+ filename +".pdf", dpi = 600)
    plt.show()
    plt.close()
    print(comp_inds[0])
    print("max error:{}".format(max(LD_e)))
    print("min error:{}".format(min(LD_e)))
    print("average error:{}".format(aver_e))

    ##### save the results #####
    # fo.write(filename)
    # fo.write("max error:{}".format(max(LD_e)))
    # fo.write("min error:{}".format(min(LD_e)))
    # fo.write("average error:{}".format(aver_e))
    # fo.write("\n")
    # fo.close()

