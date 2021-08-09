from numpy import tri
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

def visualize_3d_generated_trj(val_trj, trj_y, device, fig_number=1,):
    n_trj = len(val_trj)
    dim = val_trj[0].shape[-1]

    plt.figure(fig_number).clf()
    fig = plt.figure(num=fig_number)
    ax = fig.gca(projection='3d')
    ax.set_title("3D")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    colors = ["r",'g','k','b']
    for i in range(n_trj):
        # y_0 = torch.from_numpy(val_trj[i][:1, :]).float().to(device)
        # trj_y = iflow.generate_trj(y_0, T=int(val_trj[i].shape[0]*2))
        # trj_y = trj_y.detach().cpu().numpy()
        
        # print(trj_y[0:10,:])
        # print(trj_y[-10:,:])
        # print(val_trj[i][-10:,:])

        ax.plot(trj_y[i][:,0], trj_y[i][:,1], trj_y[i][:,2],color = colors[i])
        ax.plot(val_trj[i][:,0], val_trj[i][:,1], val_trj[i][:,2],color = colors[i],linestyle = ":")
        print("generated points for {}th trajectory:{}".format(i+1,len(trj_y[i])))
    #plt.savefig("/home/walter/DL/iflow/results/lasa_3d/epoch{}.jpg".format(epoch))
    plt.show()
    plt.pause(0.001)