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
    
    layers = 11
    activation_function = "ReLu"
    # fo = open(os.getcwd() +"/results/error/error.txt","w")

    # for filename in os.listdir(os.getcwd()+ '/data/LASA_HandWriting_SPD/'):  #choose input data
        # (filename, extension) = os.path.splitext(filename)
    filename = 'Sine_SPD'
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

    data = lasa_spd_dataset.LASA_SPD(filename=filename, device=device)
    dim = data.dim
    dynamics = model.TanhStochasticDynamics(dim, device=device, dt=0.003, T_to_stable=3)
    flow = create_flow_seq(dim, layers, activation_function)
    iflow = model.ContinuousDynamicFlow(dynamics=dynamics, model=flow, dim=dim).to(device)
    iflow.load_state_dict(torch.load(os.getcwd() + "/best_models/"+ filename + "_11_best.pt"))
    # dataloader = DataLoader(data.dataset, batch_size=1)
    with torch.no_grad():
        iflow.eval()
        # for local_x, local_y in dataloader:
        #     dataloader.dataset.step = 1
        #     y0 = local_x
        #     y1 = local_y[0]
        #     x_0, log_det_J_x0 = iflow(y0)
        #     x_1, log_det_J_x1 = iflow(y1)
        #     print(x_0,x_1)

        predicted_trajs = []
        for trj in data.train_data:
            # n_trj = trj.shape[0]
            y0 = trj[0, :]
            y0 = torch.from_numpy(y0[None, :]).float().to(device)
            yn = trj[-1, :]
            yn = torch.from_numpy(yn[None, :]).float().to(device)
            traj_pred = iflow.generate_trj(y0, yn)
            traj_pred = traj_pred.detach().cpu().numpy()
            predicted_trajs.append(traj_pred)
        # visualize_3d_generated_trj(data.train_data, predicted_trajs, device, fig_name=filename)
        visualize_vector_field(data.train_data, iflow, device, fig_name=filename)

        #### random satrt
        # center = data.train_data[0][0,:]    #1*3
        # max_cor = center + 0.5
        # min_cor = center - 0.5
        # n_sample = 2
        # xr = np.linspace(min_cor[0], max_cor[0], n_sample)
        # yr = np.linspace(min_cor[1], max_cor[1], n_sample)
        # zr = np.linspace(min_cor[2], max_cor[2], n_sample)

        # xyz = np.meshgrid(xr,yr,zr)
        # hvd = np.reshape(xyz,(3,n_sample**3)).T   #8*3
        # yn = data.train_data[0][-1, :]
        # yn = torch.from_numpy(yn[None, :]).float().to(device)
        # predicted_trajs = []
        # for i in range(hvd.shape[0]):
        #     y0 = hvd[i, :]
        #     y0 = torch.from_numpy(y0[None, :]).float().to(device)
        #     traj_pred = iflow.generate_trj(y0, yn)
        #     traj_pred = traj_pred.detach().cpu().numpy()
        #     predicted_trajs.append(traj_pred)
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # for tri in predicted_trajs:
        #     ax.plot(tri[:,0], tri[:,1], tri[:,2],color = "b", linestyle = "-")
        # for tri in data.train_data:
        #     ax.plot(tri[:,0], tri[:,1], tri[:,2],color = "g")
        # ax.view_init(elev = 10, azim = 11)
        # plt.savefig(os.getcwd() +"/results/stream/stream_3d.svg")
        # plt.show()

    '''
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
    for i in range(4):
        comp_inds.append(find_pair(predicted_trajs[i], data.train_data[i]))
        for j in range(1000):
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
    plt.autoscale(enable=True, axis='both', tight=None)
    plt.savefig(os.getcwd() +"/results/error/"+ filename +".svg")
    # plt.show()
    # print(comp_inds[0])
    print("max error:{}".format(max(LD_e)))
    print("min error:{}".format(min(LD_e)))
    print("average error:{}".format(aver_e))
#     fo.write(filename)
#     fo.write("max error:{}".format(max(LD_e)))
#     fo.write("min error:{}".format(min(LD_e)))
#     fo.write("average error:{}".format(aver_e))
#     fo.write("\n")
# fo.close()
'''
