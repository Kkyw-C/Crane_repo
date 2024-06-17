# -*- coding: UTF-8 -*-
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from random import randint
from planning_data_visual_tool import PlanningDataVisualTool
from models import Obs2QmapModel, Obs2QmapModelwithRecon

writer = SummaryWriter('runs/qmap_logs')

def generate_data(data_file):
    with open(data_file, "rb") as data_f:
        data = pickle.load(data_f)

    X = []       # 障碍物灰度图
    Y = []       # 对应的站位质量图
    Task = []       # 存储对应的任务
    for i in range(len(data)):
        task_id = "task_" + str(i)
        task_data = data[task_id]
        lift_obj_pose_init = task_data['lift_obj_pose_init']
        lift_obj_pose_goal = task_data['lift_obj_pose_goal']
        obs = task_data['obs_init']
        qmap = task_data['location_quality_map']

        X.append(obs.flatten())
        Y.append(qmap.flatten())
        Task.append([lift_obj_pose_init, lift_obj_pose_goal])

        # print(task_data)
        # vt = PlanningDataVisualTool()
        # vt.visualize_global_map('gray')
        # vt.visualize_local_map(lift_obj_pose_init,
    #plt.show()
    #plt.show() obs, 'green')
        # vt.visualize_local_map(lift_obj_pose_goal, task_data['obs_goal'], 'red')
        # vt.visualize_local_map(lift_obj_pose_init, qmap, 'blue')
        # plt.show()

    X = np.array(X)
    Y = np.array(Y)
    print("X: ", X)
    print("Y: ", Y)
    print("Task: ", Task)

    return X, Y, Task

def generate_data_1(data_file):
    with open(data_file, "rb") as data_f:
        data = pickle.load(data_f)

    X = []       # 障碍物灰度图
    Y = []       # 对应的站位质量图
    Z = []
    Lift_Obj_Poses = []       # 存储对应的任务
    num = len(data)
    for i in range(num):
        task_id = "lift_obj_pose_" + str(i)
        task_data = data[task_id]
        lift_obj_pose = task_data['lift_obj_pose']
        obs = task_data['obs']
        qmap = task_data['location_quality_map']

        # X.append(obs.flatten())
        # Y.append(qmap.flatten())
        # Lift_Obj_Poses.append([lift_obj_pose])
        Z.append(np.concatenate([lift_obj_pose, obs.flatten(), qmap.flatten()], np.newaxis))

        # #print(task_data)
        # vt = PlanningDataVisualTool()
        # vt.visualize_global_map('gray')
        # vt.visualize_local_map(lift_obj_pose, obs, 'green')
        # vt.visualize_local_map(lift_obj_pose, qmap, 'blue')
        # plt.show()

    Z = np.array(Z)
    print(Z.shape)
    #np.random.shuffle(Z)
    Lift_Obj_Poses = Z[:, 0 : 3]
    X = Z[:, 3: 3+56*56]
    Y = Z[:, 3+56*56 : 3+2*56*56]
    print(Lift_Obj_Poses.shape)
    print(X.shape)
    print(Y.shape)
    num_train = int(0.9 * num)
    X_train = X[0:num_train, :]
    Y_train = Y[0:num_train, :]
    Lift_Obj_Poses_train = Lift_Obj_Poses[0:num_train, :]
    X_test = X[num_train:num, :]
    Y_test = Y[num_train:num, :]
    Lift_Obj_Poses_test = Lift_Obj_Poses[num_train:num, :]
    # print("X: ", X)
    # print("Y: ", Y)
    # print("Task: ", Lift_Obj_Poses)

    return X_train, Y_train, Lift_Obj_Poses_train, X_test, Y_test, Lift_Obj_Poses_test

data_file = "/home/lys/catkin_ws/src/crawler_crane/crane_tutorials/crane_planning_data/data_Qmap.pkl"
X_train, Y_train, Lift_Obj_Poses_train, X_test, Y_test, Lift_Obj_Poses_test = generate_data_1(data_file)


# data_file = "/home/lys/catkin_ws/src/crawler_crane/crane_tutorials/crane_planning_data/test_data_Qmap.pkl"
# X_test, Y_test, test_Task = generate_data(data_file)

def train(test_Task, X, Y, prefix, fine_tune = False):
    obs_dim = 56 * 56
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # cvae = Obs2QmapModel(
    #     obs_dim=56 * 56,
    #     qmap_dim=56 * 56
    # ).to(device)

    cvae = Obs2QmapModelwithRecon(obs_dim=56*56, qmap_dim=56*56)

    print(cvae)

    # if fine_tune == True:
    #     cvae.load_state_dict(torch.load('th_models/init/Obs2QmapModel-109999.pt'))

    optimizer = torch.optim.Adam(cvae.parameters(), lr=10e-6)

    it = 0
    numTrain = len(X)
    mb_size = 16


    #fp = open("train_qmap_log.txt", 'a', encoding='utf-8')
    running_loss = 0.0
    for it in range(it, it + 500001):
        # randomly generate batches
        batch_elements = [randint(1, numTrain - 1) for n in range(0, mb_size)]
        #print(X)
        X_mb = X[batch_elements, :]
        Y_mb = Y[batch_elements, :]

        X_mb = torch.from_numpy(X_mb).float().to(device)
        Y_mb = torch.from_numpy(Y_mb).float().to(device)

        # qmap = cvae(X_mb)
        # loss = cvae.loss(qmap, Y_mb)
        qmap, recon_obs, mean, log_var = cvae(X_mb)
        loss = cvae.loss(qmap, Y_mb, recon_obs, X_mb, mean, log_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it == 0:
            writer.add_scalar('training loss', loss.item(), it)
        running_loss += loss.item()
        if it % 500 == 499:
            print('Iter: {}'.format(it))
            print('Loss: {:.4}'.format(loss))
            print()
            writer.add_scalar('training loss', running_loss / 500.0, it)
            running_loss = 0.0

            numTest = len(X_test)
            idx = randint(0, numTest-1)
            x = torch.from_numpy(X_test[idx]).float()
            y_target = torch.from_numpy(Y_test[idx])
            y, recon_x, mean, log_var = cvae.forward(x)
            x = x.detach()
            recon_x.detach()
            # print(x.shape)
            # print(recon_x.shape)
            # print(y.shape)
            y_target = y_target.detach()
            y = y.detach()
            # yt = np.reshape(y_target, (56, 56))
            # yy = [int(y[i]) for i in range(len(y))]
            # yy = np.reshape(yy, (56, 56))
            # fp.write("y_target: \n" + str(yt.numpy().tolist()))
            # fp.write("\n")
            # fp.write("y: \n" + str(yy.tolist()))

            #show_distribution(test_Task[idx], x.numpy(), y_target.numpy(), y.numpy(), it, prefix)
            show_distribution_with_recon(test_Task[idx], x.numpy(), recon_x.detach().numpy(), y_target.numpy(), y.numpy(), it, prefix)
            torch.save(cvae.state_dict(), os.path.join("th_models", prefix, "Obs2QmapModel-{:d}.pt".format(it)))

vt = PlanningDataVisualTool()
def show_distribution(lift_obj_pose, x, y_target, y, it, prefix):
    plt.axes().set_aspect('equal')  # 为了绘制的图形不变形
    plt.title('Lifting Scene')
    plt.xlabel('X')
    plt.ylabel('Y')

    vt.visualize_global_map('gray')


    x = np.reshape(x, (56, 56))
    vt.visualize_local_map(lift_obj_pose, x, 'blue', flag=True)

    y_target = np.reshape(y_target, (56, 56))
    vt.visualize_local_map(lift_obj_pose, y_target, 'green', flag=True)

    y = np.reshape(y, (56, 56))
    vt.visualize_local_map(lift_obj_pose, y, 'red', flag=True)

    plt.savefig("figs/{:s}/qmap-fig-{:d}.png".format(prefix, it), dpi=300)

    # plt.show()

    plt.clf()
    plt.close()

def show_distribution_with_recon(lift_obj_pose, x, recon_x, y_target, y, it, prefix):
    plt.axes().set_aspect('equal')  # 为了绘制的图形不变形
    plt.title('Lifting Scene')
    plt.xlabel('X')
    plt.ylabel('Y')

    vt.visualize_global_map('gray')


    x = np.reshape(x, (56, 56))
    vt.visualize_local_map(lift_obj_pose, x, 'blue', flag=True)

    recon_x = np.reshape(recon_x, (56, 56))
    vt.visualize_local_map(lift_obj_pose, recon_x, 'yellow', flag=True)

    y_target = np.reshape(y_target, (56, 56))
    vt.visualize_local_map(lift_obj_pose, y_target, 'green', flag=True)

    y = np.reshape(y, (56, 56))
    vt.visualize_local_map(lift_obj_pose, y, 'red', flag=True)

    plt.savefig("figs/{:s}/qmap-fig-{:d}.png".format(prefix, it), dpi=300)

    # plt.show()

    plt.clf()
    plt.close()

def show():

    n = len(X_test)
    for i in range(n):
        vt.visualize_global_map('gray')
        x = np.reshape(X_test[i], (56, 56))
        y = np.reshape(Y_test[i], (56, 56))
        vt.visualize_local_map(Lift_Obj_Poses_test[i], x, 'green')
        vt.visualize_local_map(Lift_Obj_Poses_test[i], y, 'red')
        plt.show()


#show()

train(Lift_Obj_Poses_test, X_train, Y_train, 'init', fine_tune=True)

# data_file = "/home/lys/catkin_ws/src/crawler_crane/crane_tutorials/crane_planning_data/planning_data_Qmap.pkl"
# X, Y, Task = generate_data(data_file)
# train(Task, X, Y, 'init')
