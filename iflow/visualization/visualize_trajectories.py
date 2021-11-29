from numpy import tri
import torch
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plane_coord(xr,yr,zr,a,b,c,d):
    '''
    generate the coordinates of a plane (ax+by+cy+d=0) in 3d space of xr range alone x axis, same for y and z
    only 2 of xr,yr,zr are lists, the other one must be []
    a,b,c,d are scalars that all should be given 
    '''
    coord = []
    for x in xr:
        for y in yr:
            coord.append((x,y,x))
    return coord


def visualize_trajectories(val_trajs, iflow, device, fig_number=1):
    dim = val_trajs[0].shape[1]

    plt.figure(fig_number, figsize=(20, int(10 * dim))).clf()
    fig, axs = plt.subplots(dim, 1, num=fig_number)

    for trj in val_trajs:
        n_trj = trj.shape[0]
        y0 = trj[0, :]
        y0 = torch.from_numpy(y0[None, :]).float().to(device)
        traj_pred = iflow.generate_trj( y0, T=n_trj)
        traj_pred = traj_pred.detach().cpu().numpy()

        for j in range(dim):
            axs[j].plot(trj[:,j],'b')
            axs[j].plot(traj_pred[:,j],'r')
    plt.draw()
    plt.pause(0.001)


def visualize_2d_generated_trj(val_trj, iflow, device, fig_number=1):
    n_trj = len(val_trj)
    dim = val_trj[0].shape[-1]

    plt.figure(fig_number).clf()
    fig = plt.figure(figsize=(15, 15), num=fig_number)
    for i in range(len(val_trj)):
        y_0 = torch.from_numpy(val_trj[i][:1, :]).float().to(device)
        trj_y = iflow.generate_trj(y_0, T=val_trj[i].shape[0])
        trj_y = trj_y.detach().cpu().numpy()

        plt.plot(trj_y[:,0], trj_y[:,1], 'g')
        plt.plot(val_trj[i][:,0], val_trj[i][:,1], 'b')
    plt.draw()
    plt.pause(0.001)

def visualize_3d_generated_trj(val_trj, trj_y, device, fig_number=1, fig_name = None):
    '''
    Max number of demonstration trajectories: 7

    if the dimention of the data is 3, the function will plot the demonstration(dot line) and the prediction(solid line),
    otherwise the function will plot the last 3 dimentions of the data

    if the number of prediction trajectories are larger than the number of demonstration trajectories,
    i.e. user give new strating points to generated new trajectories, the function will also show them in the same figure with big dot
    '''
    n_trj = len(val_trj)
    dim = val_trj[0].shape[-1]

    plt.figure(fig_number).clf()
    fig = plt.figure(num=fig_number)
    ax = fig.gca(projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    colors = ["k",'g','r','b', 'c', 'y', 'm']
    for i in range(n_trj):
        ax.plot(trj_y[i][:,-3], trj_y[i][:,-2], trj_y[i][:,-1],color = colors[i])
        ax.plot(val_trj[i][:,-3], val_trj[i][:,-2], val_trj[i][:,-1],color = colors[i],linestyle = ":")
    for i in range(n_trj, len(trj_y)):
        ax.plot(trj_y[i][:,-3], trj_y[i][:,-2], trj_y[i][:,-1], "o")
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.view_init(elev=100, azim=-90)
    # plt.savefig(os.getcwd() +"/results/trajectories/"+ fig_name +".svg", dpi = 600)
    # plt.close()
    plt.show()
    # plt.pause(0.001)