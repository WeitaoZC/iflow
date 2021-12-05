#coding=utf-8
'''
Script processing results from the well-trained model for artificialy made uq data
'''
from operator import ne
import os, torch
from numpy.lib.function_base import average, select
import numpy as np
import scipy.io as spio
from scipy.linalg import expm, logm
import scipy.interpolate as ci
from iflow import model
from iflow.dataset import lasa_uq_dataset
from iflow.visualization import visualize_3d_generated_trj
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from scipy.spatial.transform import Rotation as R

device = torch.device('cuda:' + str(0) if torch.cuda.is_available() else 'cpu')
directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '../', 'data')) + '/LASA_HandWriting_UQ/'


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

def quat_conj(q):
    '''quaternion conjugation'''
    q[1:] *= -1
    return q

def quat_prod(q1,q2):
    '''quaternion product'''
    p = np.zeros(4)
    p[0] = q1[0] * q2[0] - q1[1:] @ q2[1:]
    p[1:] = q1[0] * q2[1:] + q2[0] * q1[1:] + np.cross(q1[1:],q2[1:])
    return p

def quat_Log(q):
    '''Logorithm map: manifold to tangent space
    one uq to one vector
    '''
    nEpsilon = np.linalg.norm(q[1:])
    qlog = np.arccos(q[0])*q[1:]/nEpsilon
    return qlog

def quat_log_err(q1,q2):
    '''uq comparison (L2 distance in the tangent space)'''
    err = 2*quat_Log(quat_prod(q1, quat_conj(q2)))
    err = np.linalg.norm(err)
    return err

# rotate vector v1 by quaternion q1 
def qv_mult(q1, v1):
    # comment this out if v1 doesn't need to be a unit vector
    q2 = list(v1)
    q2.append(0.0)
    q2 = np.array(q2)
    return quat_prod(quat_prod(q1, q2), quat_conj(q1))[:3]

def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return np.array([-x1*x0 - y1*y0 - z1*z0 + w1*w0,
                        x1*w0 + y1*z0 - z1*y0 + w1*x0,
                        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
                        x1*y0 - y1*x0 + z1*w0 + w1*z0])

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
    ##### Must keep the same parameter with the used model parameter file #####
    filename = 'Sshape'
    layers = 11
    activation_function = "ReLu"

    ##### load original data to acquire demonstration data #####
    data = spio.loadmat(directory + filename + '_UQ.mat', squeeze_me=True)
    UQs = []
    for demo_i in data['demoUQ']:
        UQs.append(demo_i.tolist()[3].transpose()) #4*1000*4
    UQs = np.array(UQs)

    ##### load data for model and build the corresponding model for saved model file #####
    data = lasa_uq_dataset.LASA_UQ(filename=filename + "_UQ", device=device)
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
            n_trj = trj.shape[0]
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

    ##### use exp map transferring to the manifold #####
    gener_uqs = []
    LD_e = []
    for i in range(4):
        gener_uqs.append(R_Exp(unnormlized[i]))

    ##### find corresponding pairs from generated data and demonstration to compare #####
    comp_inds = []
    for i in range(len(predicted_trajs)):
        comp_inds.append(find_pair(predicted_trajs[i], data.train_data[i]))
        for j in range(len(comp_inds[i])):
                LD_e.append(quat_log_err(gener_uqs[i][comp_inds[i][j]],UQs[i,j]))
    aver_e = sum(LD_e) / 4000

    ##### check computations #####
    # print("choice:{}".format(comp_inds[0][500]))
    # print("generated:{}".format(predicted_trajs[0][comp_inds[0][500],:]))
    # print("unnormed:{}".format(unnormlized[0][comp_inds[0][500],:]))
    # print("generaed uq:{}".format(gener_uqs[0][comp_inds[0][500],:]))
    # print("given uq:{}".format(UQs[0, 500]))

    ##### select chosen point from generated data #####
    selected_uqs = []
    for i in range(4):
        selected_uqs.append(gener_uqs[i][comp_inds[i]])
    selected_uqs = np.array(selected_uqs)
    selected_uqs[:,:,1:] = -selected_uqs[:,:,1:]

    ##### save the distance of uq pairs to the list #####
    # LD_e = []
    # for i in range(len(selected_uqs)):
    #     for j in range(len(selected_uqs[i])):
    #         LD_e.append(quat_log_err(selected_uqs[i][j], UQs[i,j]))

    ##### rotate a point using the quaternion#####
    pred_points = []
    demo_points = []
    point = [1,0,0]
    for i in range(len(selected_uqs)):
        points1 = np.zeros((selected_uqs[i].shape[0], 3))
        points2 = np.zeros((selected_uqs[i].shape[0], 3))
        for j in range(selected_uqs[i].shape[0]):
            r1 = R.from_quat(selected_uqs[i][j,:])
            points1[j,:] = r1.apply(point)
            r2 = R.from_quat(UQs[i,j,:])
            points2[j,:] = r2.apply(point)
        pred_points.append(points1)
        demo_points.append(points2)
    print(np.linalg.norm(pred_points[1][100,:]))
    print(pred_points[0].shape)
    print(demo_points[0].shape)



    ##### draw uqs on the sphere #####
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    t = np.linspace(0, np.pi * 2, 200)
    s = np.linspace(0, np.pi, 200)

    t, s = np.meshgrid(t, s)
    x = np.cos(t) * np.sin(s)
    y = np.sin(t) * np.sin(s)
    z = np.cos(s)
    #x = 1 * np.outer(np.cos(t), np.sin(s))
    #y = 1 * np.outer(np.sin(t), np.sin(s))
    #z = 1 * np.outer(np.ones(np.size(t)), np.cos(s))

    ax = plt.subplot(111, projection='3d')

    ax.plot_surface(x, y, z,  rstride=10, cstride=10, cmap='gray', edgecolors='k', linewidth = 0, alpha=.2)
    #ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    #ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    #ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    #ax.plot_wireframe(x, y, z, color="r")
    ax.grid(False)
    gene_color = ["grey", "springgreen", 'tomato', 'deepskyblue']
    demo_color = ["k","g","r","b"]
    for i in range(UQs.shape[0]):
        # ax.plot(selected_uqs[i,:,0], selected_uqs[i,:,1], selected_uqs[i,:,3],color = gene_color[i], linewidth = 2.5)
        # ax.plot(UQs[i,:,0], UQs[i,:,1], UQs[i,:,3], demo_color[i], linestyle = ":", linewidth = 2.5)
        ax.plot(pred_points[i][:,2], pred_points[i][:,1], pred_points[i][:,0],color = gene_color[i], linewidth = 2.5)
        ax.plot(demo_points[i][:,2], demo_points[i][:,1], demo_points[i][:,0], demo_color[i], linestyle = ":", linewidth = 2.5)
    ax.view_init(elev=0, azim=0)
    ax.grid(False)
    plt.axis('off')
    # plt.savefig("C:/Users/Walter/Desktop/research/flow based/figure/UQsphere.pdf", dpi=600)
    plt.show()
    plt.close()

'''
    ##### plot the comparison results (distance of SPD pairs)#####
    # plt.figure(figsize=(10, 5))
    # plt.plot([i for i in range(1000)], LD_e[0:1000], "k")
    # plt.plot([i for i in range(1000)], LD_e[1000:2000], "g")
    # plt.plot([i for i in range(1000)], LD_e[2000:3000], "r")
    # plt.plot([i for i in range(1000)], LD_e[3000:4000], "b")
    # plt.ylabel("Log quaternion distanc")
    # # plt.savefig("C:/Users/Walter/Desktop/research/flow based/figure/UQerror.pdf", dpi=600)
    # plt.show()
    # plt.close()
    # print("max error:{}".format(max(LD_e)))
    # print("min error:{}".format(min(LD_e)))
    # print("average error for {} :{}".format(filename, aver_e))
'''

