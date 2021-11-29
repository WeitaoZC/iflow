from matplotlib.colors import Normalize, same_color
import torch
import numpy as np
import os
from iflow.utils.generic import to_numpy, to_torch
import matplotlib.pyplot as plt


def visualize_latent_distribution(val_trj, iflow, device, fig_number=1):
    n_trj = len(val_trj)
    dim = val_trj[0].shape[-1]
    ## Store the latent trajectory in a list ##
    val_z_trj= []
    val_mu_trj = []
    val_var_trj = []

    plt.figure(fig_number, figsize=(20,int(10*dim))).clf()
    fig, axs = plt.subplots(dim, 1, num=fig_number)
    for i in range(len(val_trj)):
        y_trj = to_torch(val_trj[i],device)
        z_trj, _ = iflow(y_trj)
        z_trj = to_numpy(z_trj)
        val_z_trj.append(z_trj)

        z0 = to_torch(z_trj[0,:],device)
        trj_mu, trj_var = iflow.dynamics.generate_trj_density(z0[None,:], T = val_z_trj[i].shape[0])
        val_mu_trj.append(to_numpy(trj_mu))
        val_var_trj.append(to_numpy(trj_var))

        for j in range(val_trj[i].shape[-1]):
            t = np.linspace(0,val_z_trj[i].shape[0], val_z_trj[i].shape[0])
            axs[j].plot(t,val_z_trj[i][:,j])
            l_trj = val_mu_trj[i][:,0,j] - np.sqrt(val_var_trj[i][:,0, j, j] )
            h_trj = val_mu_trj[i][:,0,j]  + np.sqrt(val_var_trj[i][:,0, j, j] )
            axs[j].fill_between(t,l_trj, h_trj, alpha=0.1)

    plt.draw()
    plt.pause(0.001)


def visualize_vector_field(val_trj, iflow, trj_y, device, fig_number=1, fig_name = None):
    '''
    stream plot for 3d space
    since there is no buildin function to implement 3d stream figure,
    here choose one plane (or project the 3d space to 2d plane) to plot the stream figure using buildin function
    and extract the stream lines of the 2d plane replot them in the 3d sapce 
    '''
    n_trj = len(val_trj)
    _trajs = np.zeros((0, 3))
    for trj in val_trj:
        _trajs = np.concatenate((_trajs, trj), 0)  # 4000*3
    ##### find the board of the figure to generate grid #####
    min = _trajs.min(0) - 0.5
    max = _trajs.max(0) + 0.5
    n_sample = 100
    x = np.linspace(min[0], max[0], n_sample)  # 100
    y = np.linspace(min[1], max[1], n_sample)
    z = np.linspace(min[2], max[2], n_sample)
    xyz = np.meshgrid(x, y, z)  # 3*100*100*100
    hvd = torch.Tensor(np.reshape(xyz, (3, n_sample**3)).T).float()

    if device is not None:
        hvd = hvd.to(device)
    ##### using the model generating next points for each point of the space grid #####
    hvd_t1 = iflow.evolve(hvd, T=3)
    hvd = hvd.detach().cpu().numpy()
    hvd_t1 = hvd_t1.detach().cpu().numpy()

    ##### direction vector at each grid point #####
    vel = (hvd_t1 - hvd)
    vel_x = np.reshape(vel[:, 0], (n_sample, n_sample, n_sample))
    vel_y = np.reshape(vel[:, 1], (n_sample, n_sample, n_sample))
    vel_z = np.reshape(vel[:, 2], (n_sample, n_sample, n_sample))
    speed = np.sqrt(vel_x ** 2 + vel_y ** 2 + vel_z ** 2)
    speed = speed/np.max(speed)

    ##### save for matlab #####
    # mat_path = os.getcwd() + '/results/mat' + fig_name + '.mat'
    # io.(mat_path, {'x': xyz[0], 'y': xyz[1],
    #            'z': xyz[2], 'u': vel_x, 'v': vel_y, 'w': vel_z})

    ##### choose one plan to plot the 2d stream plot #####
    plt.clf()
    ax_tem = plt.gca()
    # vel_x = np.sum(vel_x, axis=2)/n_sample
    # vel_y = np.sum(vel_y, axis=2)/n_sample
    # vel_z = np.sum(vel_z, axis=2)/n_sample
    speed = np.sum(speed, axis=2)/n_sample
    vel_x_2d = np.zeros((n_sample, n_sample))
    vel_y_2d = np.zeros((n_sample, n_sample))
    for i in range(n_sample):
        vel_x_2d[:, i] = vel_x[:, i, i]
        vel_y_2d[:, i] = vel_y[:, i, i]
    res = ax_tem.streamplot(xyz[0][:, :, 99], xyz[1][:, :, 99],
                            vel_x_2d, vel_y_2d, color=speed, density=[0.5, 1])
    ##### extract the 2d stream lines #####
    lines = res.lines.get_paths()

    ##### replot the 2d stream lines in the 3d space #####
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_xlabel("\$x$")
    ax.set_ylabel("\$y$")
    ax.set_zlabel("\$z$")
    for line in lines:
        old_x = line.vertices.T[0]
        old_y = line.vertices.T[1]
        ##### apply 2d to 3d transformation here #####
        new_z = old_x
        new_x = old_x
        new_y = old_y
        ##### plot stream lines #####
        ax.plot(new_x, new_y, new_z,"c", linewidth = 1)

    ##### plot generated and demonstration trajectories #####
    colors = ["k",'g','r','b']
    for i in range(n_trj):
        ax.plot(trj_y[i][:,0], trj_y[i][:,1], trj_y[i][:,2],color = colors[i], linewidth = 2)
        ax.plot(val_trj[i][:,0], val_trj[i][:,1], val_trj[i][:,2],color = colors[i],linestyle = ":", linewidth = 2)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.view_init(elev=100, azim=-90)
    # plt.savefig(os.getcwd() + "/results/stream3d/100dpi/" +
    #             fig_name + ".pdf", dpi=100)
    plt.show()
    plt.close()
    

def save_vector_field(val_trj, iflow, device, save_fig, fig_number=1):
    _trajs = np.zeros((0, 2))
    for trj in val_trj:
        _trajs = np.concatenate((_trajs, trj),0)
    min = _trajs.min(0) - 0.5
    max = _trajs.max(0) + 0.5

    n_sample = 100

    x = np.linspace(min[0], max[0], n_sample)
    y = np.linspace(min[1], max[1], n_sample)

    xy = np.meshgrid(x, y)
    h = np.concatenate(xy[0])
    v = np.concatenate(xy[1])
    hv = torch.Tensor(np.stack([h, v]).T).float()
    if device is not None:
        hv = hv.to(device)

    hv_t1 = iflow.evolve(hv, T=3)
    hv = hv.detach().cpu().numpy()
    hv_t1 = hv_t1.detach().cpu().numpy()

    vel = (hv_t1 - hv)

    vel_x = np.reshape(vel[:, 0], (n_sample, n_sample))
    vel_y = np.reshape(vel[:, 1], (n_sample, n_sample))
    speed = np.sqrt(vel_x ** 2 + vel_y ** 2)
    speed = speed/np.max(speed)

    fig = plt.figure(fig_number, figsize=(10, 10))
    plt.clf()
    ax = plt.gca()

    plt.streamplot(xy[0], xy[1], vel_x, vel_y, color=speed, density=[0.5, 1])
    for i in range(len(val_trj)):
        plt.plot(val_trj[i][:,0], val_trj[i][:,1], 'b')

    plt.savefig(save_fig)
