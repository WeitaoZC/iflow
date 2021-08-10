import os, torch
from numpy.lib.function_base import average
import numpy as np
import scipy.io as spio
from scipy.linalg import expm, logm
import scipy.interpolate as ci
from iflow import model
from iflow.dataset import lasa_3d_dataset
from iflow.visualization import visualize_3d_generated_trj
import matplotlib.pyplot as plt

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'data')) + '/LASA_HandWriting_SPD/'


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


def vec2sym_mat(V):  # 4*1000*3
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
                m = m + np.diag(v[ind[k]:], k + 1) / np.sqrt(2) + np.diag(v[ind[k]:], -k - 1) / np.sqrt(
                    2)  # Mandel notation
            M.append(m)

    return M


def exp_map(U):
    '''
    exponential map: tangent space to manifold
    symmetric matrix to SPD

    Note: because of matrix power is only provided for int
    this func only compute for the goal point's tangent space
    '''
    P = np.array([[10, 0], [0, 10]])
    N = np.array([[0.1, 0], [0, 0.1]])
    m = P @ expm(N @ U @ N) @ P

    return m


def Log_Euclidean_d(A, B):
    res = np.linalg.norm(logm(B) - logm(A), ord='fro')
    return res


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
    filename = 'NShape_SPD'
    layers = 8
    activation_function = "ReLu"
    data = spio.loadmat(directory + filename + '.mat', squeeze_me=True)
    SPDs = []
    for demo_i in data['demoSPD']:
        SPDs.append(demo_i.tolist()[3])

    SPDs = np.array(SPDs)
    SPD_m = []
    for j in range(SPDs.shape[0]):
        for i in range(SPDs.shape[-1]):
            m = np.array([SPDs[j][0][0][i], SPDs[j][0][1][i], SPDs[j][1][0][i], SPDs[j][1][1][i]]).reshape(2, 2)
            SPD_m.append(m)

    data = lasa_3d_dataset.LASA3D(filename=filename, device=device)
    dim = data.dim
    dynamics = model.TanhStochasticDynamics(dim, device=device, dt=0.003, T_to_stable=3)
    flow = create_flow_seq(dim, layers, activation_function)
    iflow = model.ContinuousDynamicFlow(dynamics=dynamics, model=flow, dim=dim).to(device)
    iflow.load_state_dict(torch.load(os.getcwd() + "/results/lasa_3d/normal/saved_model/" + filename + ".pt"))
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
    sym_m = vec2sym_mat(unnormlized)

    # print("***first point****")
    # print(unnormlized[0][0])
    # print(sym_m[0])
    # SPD1 = exp_map(sym_m[0])
    # print(SPD1)
    # print(SPD_m[0])
    # d = Log_Euclidean_d(SPD1,SPD_m[0])
    # print(d)

    # print("***last point****")
    # print(unnormlized[0][-1])
    # SPD1 = exp_map(sym_m[-1])
    # print(SPD1)
    # print(SPD_m[-1])
    # d = Log_Euclidean_d(SPD1,SPD_m[-1])
    # print(d)

    comp_inds = []
    LD_e = []
    begin = 0
    for i in range(len(unnormlized)):
        comp_inds.append(find_pair(predicted_trajs[i], data.train_data[i]))
        for j in range(len(comp_inds[i])):
            LD_e.append(Log_Euclidean_d(exp_map(sym_m[begin + comp_inds[i][j]]), SPD_m[i * 1000 + j]))
            # print(begin+comp_inds[i][j])
        begin += len(unnormlized[i])
    aver_e = sum(LD_e) / 4000

    # fig = plt.figure()
    # ax1 = plt.axes(projection='3d')
    # ax1.plot3D(data.train_data[0][:,0],data.train_data[0][:,1],data.train_data[0][:,2],'red')    #绘制空间曲线

    #### interpolate for given trajectories to match generated trajectories (did not fit well)
    # f = ci.LinearNDInterpolator(list(zip(data.train_data[0][:,1], data.train_data[0][:,2])), data.train_data[0][:,0])
    # f = ci.Rbf(data.train_data[0][0,:], data.train_data[0][1,:], data.train_data[0][2,:],function = 'multiquadric')
    # xnew = predicted_trajs[0][comp_inds[0],0]
    # ynew = predicted_trajs[0][comp_inds[0],1]
    # znew = f(xnew, ynew)
    # print(len(znew))
    # ax1.plot3D(xnew,ynew,znew,'b')
    # plt.show()

    plt.plot([i for i in range(1000)], LD_e[0:1000], "r")
    plt.plot([i for i in range(1000)], LD_e[1000:2000], "g")
    plt.plot([i for i in range(1000)], LD_e[2000:3000], "k")
    plt.plot([i for i in range(1000)], LD_e[3000:4000], "b")
    plt.show()
    print(comp_inds[0])
    print("max error:{}".format(max(LD_e)))
    print("min error:{}".format(min(LD_e)))
    print("average error for {} :{}".format(filename, aver_e))
