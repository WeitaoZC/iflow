#coding=utf-8
'''
Script processing raw data acquired from real robot experiment (position and oritation data(uqs))
'''
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.utils import data

##### goal #####
goal_x = 0.6768
goal_y = -0.04434
goal_z = 0.2189

goal_ori1 = 0.01671  #-0.00715
goal_ori2 = 0.9911
goal_ori3 = -0.1321
goal_ori4 = 0.0139

##### threshold for split function #####
yita1 = 0 #0.00001
yita2 = 0.00005

##### load raw data #####
realtrial = np.loadtxt(os.getcwd() + "/data/Robot_data/realTrial.txt")

def split(data, goal, yita1, yita2):
    '''
    split the data according to the goal
    acquire several demonstrations from the raw data
    '''
    dataset = []
    num = data.shape[0]
    i = j = 0
    while i<num:
        if ((data[i,-5] - goal) < yita1):
            j = i-4000
            while (data[j,-5]<0.4):
                j-=500
            while j>0:
                if ((data[j-1,-5] - data[j,-5])< yita2 and (data[j-2,-5] - data[j-1,-5])< yita2):
                    dataset.append(data[j:i+1,:])
                    break
                j-=1
            i+=10000
        i+=1
    return dataset

def quat_Log(q):
    '''Logorithm map: manifold to tangent space
    one uq to one vector
    '''
    nEpsilon = np.linalg.norm(q[:,1:], axis= 1).reshape(-1,1)
    qlog = np.arccos(q[:,0]).reshape(-1, 1)*q[:,1:]/nEpsilon
    return qlog

if __name__ == '__main__':
    ##### select our interested dimention of the raw data #####
    # interest_data = realtrial[:,-7:]
    interest_data = realtrial[:,13:]

    ##### split the data to several demonstrations #####
    dataset = split(interest_data, goal_z, yita1, yita2)

    ##### check the last points of the split demonstrations #####
    # for i in range(len(dataset)):
    #     print(dataset[i][-1,7:10])
    # print("******")

    ##### save desired number of points for each demonstration #####
    for i in range(len(dataset)):
        print(dataset[i][-1,10:])
        joint_angle_pos = dataset[i][-4000:,:10]
        print(joint_angle_pos.shape)
        np.savetxt("D:/Pytorch/iflow/iflow/data/Robot_data/real_joint_pos_data/joint_angle_pos_{}.txt".format(i+1), joint_angle_pos)

    ##### if the last points do not match well with the goal use the following code to check and add the goal manually #####
'''   
    pos_data = []
    for i in range(len(dataset)):
        pos_data.append(dataset[i][-4000:,0:3])
        pos_data[i][-1,0] = goal_x
        pos_data[i][-1,1] = goal_y
        pos_data[i][-1,2] = goal_z


    # print(pos_data[0][-2,])
    # print(pos_data[1][-2,])
    # print(pos_data[2][-2,])
    # print(pos_data[3][-2,])

    ori_data = []
    for i in range(len(dataset)):
        ori_data.append(dataset[i][-4000:,3:])
        ori_data[i][-1,0] =  sum([dataset[j][-2,3]  for j in range(len(dataset))])/4 # goal_ori1
        ori_data[i][-1,1] =  goal_ori2
        ori_data[i][-1,2] =  goal_ori3
        ori_data[i][-1,3] =  goal_ori4

    for i in range(len(ori_data)):
        print(ori_data[i].shape)

    print(ori_data[0][-1,])
    # print(ori_data[1][-2,])
    # print(ori_data[2][-2,])
    # print(ori_data[3][-2,])

    # fig1 = plt.figure()
    # ax = fig1.gca(projection='3d')
    # for i in range(len(pos_data)):
    #     ax.plot(pos_data[i][:,-3], pos_data[i][:,-2], pos_data[i][:,-1], label = i)
    # plt.legend()
    # # plt.savefig(os.getcwd()+ "/data/robot_data/pos1.svg", dpi =600)

    # fig2 = plt.figure()
    # ax = fig2.gca(projection='3d')
    # for i in range(len(ori_data)):
    #     qlog = quat_Log(ori_data[i][:,-4:])
    #     ax.plot(qlog[:,-3], qlog[:,-2], qlog[:,-1], label = i)
    # plt.legend()
    # plt.show()
    # # plt.savefig(os.getcwd()+ "/data/robot_data/ori1.svg", dpi =600)

    pos = np.array(pos_data)
    print(pos.shape)
    # np.save(os.getcwd() + "/data/robot_data/pos.npy", pos)

    ori_data = np.array(ori_data)
    print(ori_data.shape)
    # np.save(os.getcwd() + "/data/robot_data/quats.npy", ori_data)

    ori_vec = []
    for i in range(len(ori_data)):
        ori_vec.append(quat_Log(ori_data[i]))
    ori_vec = np.array(ori_vec)
    print(ori_vec.shape)
    # np.save(os.getcwd() + "/data/robot_data/ori_vec.npy", ori_vec)

    ori_vec_pos = []
    for i in range(len(ori_data)):
        ori_vec_pos.append(np.hstack((ori_vec[i], pos_data[i])))
    ori_vec_pos = np.array(ori_vec_pos)
    print(ori_vec_pos.shape)
    # np.save(os.getcwd() + "/data/robot_data/real_ori_vec_pos.npy", ori_vec_pos)
'''
