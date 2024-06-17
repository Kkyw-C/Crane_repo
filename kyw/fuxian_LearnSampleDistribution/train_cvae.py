# -*- coding: UTF-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from random import randint
from planning_data_visual_tool import PlanningDataVisualTool
from models import CVAE

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def mse_loss_with_weight(x, y, w):
    se = (x-y)**2
    sew = se * w
    mse = torch.mean(sew)
    return mse

def generate_data(x_dim, data_file):
    with open(data_file, "rb") as data_f:
        data = pickle.load(data_f)

    all_x = []
    all_c = []
    x_test = []
    c_test = []
    for i in range(len(data)):
        task_id = "task_" + str(i)
        task_data = data[task_id]
        # print(task_data)
        #vt = PlanningDataVisualTool()
        #vt.visualize_global_map('gray')
        #vt.visualize_local_map(task_data['lift_obj_pose_init'], task_data['obs_init'], 'green')
        #vt.visualize_local_map(task_data['lift_obj_pose_goal'], task_data['obs_goal'], 'red')
        #plt.scatter(task_data['xytheta'][:, 0], task_data['xytheta'][:, 1], color='gray')

        # 加载x
        xytheta = task_data['xytheta']
        # 转换为相对坐标
        xytheta[:, 0] -= task_data['lift_obj_pose_init'][0]  # x
        xytheta[:, 1] -= task_data['lift_obj_pose_init'][1]  # y
        # 将样本x、y坐标转换为极角坐标
        xytheta[:, 0], xytheta[:, 1] = cart2pol(xytheta[:, 0], xytheta[:, 1])
        xytheta[:, 0] = (xytheta[:, 0] - 0.5 * (10 + 28)) / (28 - 0.5 * (10 + 28))  # 将r归一化到[-1, 1]
        xytheta[:, 1] = xytheta[:, 1] / 3.1415926  # 将alpha归一化到[-1, 1]
        print(xytheta[:, 0])
        print(xytheta[:, 1])

        if x_dim == 2:
            x = xytheta[:, 0:x_dim]
        else:
            x = xytheta
        #print("x:\n", x)
        if i == 0:
            all_x = x
        elif i == 1:
            x_test = x
            #all_x = np.concatenate([all_x, x])
        else:
            all_x = np.concatenate([all_x, x])

        # 加载被吊物目标位置和障碍物信息，作为条件c
        xytheta_goal = task_data['lift_obj_pose_goal']
        xytheta_goal[0] -= task_data['lift_obj_pose_init'][0]
        xytheta_goal[1] -= task_data['lift_obj_pose_init'][1]
        xytheta_goal = xytheta_goal[0:2]

        # vt.visualize_local_map([0, 0, 0], task_data['obs_init'], 'green')
        # vt.visualize_local_map(task_data['lift_obj_pose_goal'], task_data['obs_goal'], 'red')
        # plt.scatter([0], [0], s = 5, c = 'yellow')
        # plt.scatter([xytheta_goal[0]], [xytheta_goal[1]], s = 5, c = 'red')
        # plt.scatter(x[:, 0], x[:, 1], c = 'purple')
        # plt.show()

        obs = task_data['obs_init']# * 256  #
        c = np.concatenate([list(xytheta_goal), list(obs.flatten())], axis=0)
        n_x = len(x)
        c = np.repeat([c], n_x, axis=0)
        if i == 0:
            all_c = c
        elif i == 1:
            c_test = c
            #all_c = np.concatenate([all_c, c])
        else:
            all_c = np.concatenate([all_c, c])

    # 将数据分为训练集和测试集
    # n = len(all_x)
    # n_train = int(n * 0.8)
    # x_train = all_x[0:n_train, :]
    # c_train = all_c[0:n_train, :]
    # x_test = all_x[n_train:n, :]
    # c_test = all_c[n_train:n, :]

    x_train = all_x
    c_train = all_c

    return data, x_train, c_train, x_test, c_test


vt = PlanningDataVisualTool()
def show_distribution(obs, goal, origional_samples, recon_obs, test_samples, gen_samples, it, prefix):
    plt.axes().set_aspect('equal')  # 为了绘制的图形不变形
    plt.title('Lifting Scene')
    plt.xlabel('X')
    plt.ylabel('Y')

    # 绘制被吊物起始位置
    plt.scatter([0], [0], c='green', s=10)

    # 绘制被吊物起始位置周围局部地图
    obs = np.reshape(obs, (56, 56))
    vt.visualize_local_map([0, 0, 0], obs, 'green')

    # 绘制被吊物目标位置
    plt.scatter([goal[0]], [goal[1]], s = 10, c = 'red')

    # # 绘制原始样本
    # origional_samples[:, 0] = origional_samples[:, 0] * (28 - 0.5 * (10 + 28)) + 0.5 * (10 + 28)   # 将归一化后的值恢复回来
    # origional_samples[:, 0], origional_samples[:, 1] = pol2cart(origional_samples[:, 0], origional_samples[:, 1])   # 极坐标转换为笛卡尔坐标
    # #print(origional_samples)
    # plt.scatter(origional_samples[:, 0], origional_samples[:, 1], s = 2, c = 'gray')

    # 绘制重构样本
    recon_obs = np.reshape(recon_obs, (56, 56))
    vt.visualize_local_map([0, 0, 0], recon_obs, 'red')

    # 绘制生成样本
    test_samples[:, 0] = test_samples[:, 0] * (28 - 0.5 * (10 + 28)) + 0.5 * (10 + 28)  # 将归一化后的值恢复回来
    test_samples[:, 1] = test_samples[:, 1] * 3.1415926
    test_samples[:, 0], test_samples[:, 1] = pol2cart(test_samples[:, 0], test_samples[:, 1])  # 极坐标转换为笛卡尔坐标
    plt.scatter(test_samples[:, 0], test_samples[:, 1], s=2, c='yellow')

    # 绘制生成样本
    gen_samples[:, 0] = gen_samples[:, 0] * (28 - 0.5 * (10 + 28)) + 0.5 * (10 + 28)  # 将归一化后的值恢复回来
    gen_samples[:, 1] = gen_samples[:, 1] * 3.1415926
    gen_samples[:, 0], gen_samples[:, 1] = pol2cart(gen_samples[:, 0], gen_samples[:, 1])  # 极坐标转换为笛卡尔坐标
    plt.scatter(gen_samples[:, 0], gen_samples[:, 1], s=2, c='blue')

    sample_dim = gen_samples.shape[1]
    plt.savefig("figs/{:s}/only_mlp-fig-x{:d}-{:d}.png".format(prefix, sample_dim, it), dpi=300)

    #plt.show()

    plt.clf()
    plt.close()


def train(x_dim, x_train, c_train, x_test, c_test, prefix, only_mlp = False):
    obs_dim = 56 * 56
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def loss_fn(recon_obs, obs, mean, log_var, x, target_x, only_mlp):
        BCE = torch.nn.functional.mse_loss(
            recon_obs.view(-1, obs_dim), obs.view(-1, obs_dim), reduction='mean')
        KLD = 10 ** -6 * 2 * torch.mean(log_var.exp() + mean.pow(2) - 1. - log_var, 1)

        recon_obs_loss = torch.mean(BCE + KLD)

        # x_pred_loss = torch.nn.functional.mse_loss(
        #     x.view(-1, x_dim), target_x.view(-1, x_dim), reduction='mean')
        weight = torch.Tensor([2.0, 10.0, 0.0])
        x_pred_loss = mse_loss_with_weight(x.view(-1, x_dim), target_x.view(-1, x_dim), weight)

        if only_mlp == True:
            return x_pred_loss
        else:
            return recon_obs_loss + 10 * x_pred_loss

    cvae = CVAE(
        obs_dim=56 * 56,
        goal_dim=2,
        x_dim=3,
        latent_dim=16
    ).to(device)

    print(cvae)

    if only_mlp == True:
        cvae.load_state_dict(torch.load('th_models/goal/cvae-x3-40000.pt'))
        for k, v in cvae.named_parameters():
            if k not in ['MLP.L0.weight', 'MLP.L0.bias', 'MLP.L1.weight', 'MLP.L1.bias', 'MLP.O.weight', 'MLP.O.bias']:
                v.requires_grad = False  # 固定参数

        optimizer = torch.optim.Adam([cvae.MLP.L0.weight, cvae.MLP.L0.bias, cvae.MLP.L1.weight,cvae. MLP.L1.bias, cvae.MLP.O.weight, cvae.MLP.O.bias], lr=10e-6)
    else:
        optimizer = torch.optim.Adam(cvae.parameters(), lr=10e-6)


    it = 0
    numTrain = len(x_train)
    mb_size = 128
    for it in range(it, it + 500001):
        # randomly generate batches
        batch_elements = [randint(0, numTrain - 1) for n in range(0, mb_size)]
        X_mb = x_train[batch_elements, :]
        c_mb = c_train[batch_elements, :]
        # if (it + 1) * mb_size < numTrain:
        #     X_mb = x_train[it*mb_size : (it+1)*mb_size, :]
        #     c_mb = c_train[it * mb_size : (it + 1) * mb_size, :]
        # else:
        #     X_mb = x_train[it * mb_size : numTrain, :]
        #     c_mb = c_train[it * mb_size : numTrain, :]

        X_mb = torch.from_numpy(X_mb).float().to(device)
        c_mb = torch.from_numpy(c_mb).float().to(device)
        #print(c_mb)
        #print(X_mb.size(), c_mb.size())

        x, recon_obs, z, mean, log_var = cvae(c_mb[:, 2:56*56+2], c_mb[:, 0:2])
        #print(x.shape)
        #print(X_mb.shape)

        loss = loss_fn(recon_obs, c_mb[:, 2:56*56+2], mean, log_var, x, X_mb, only_mlp)
        if it % 500 == 0:
            # x = x.detach()
            # X_mb = X_mb.detach()
            # xx = np.array(x.numpy())
            # XX = np.array(X_mb.numpy())
            # print(xx[:, 0])
            # print(XX[:, 0])
            # # r = np.concatenate([xx[:, 0], XX[:, 0]], np.newaxis)
            # # alpha = np.concatenate([xx[:, 1], XX[:, 1]], np.newaxis)
            deta = torch.abs(x - X_mb)
            #print("deta: ", deta)
            # print("alpha: ", alpha)
            # print("loss: ", loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 500 == 0:
            print('Iter: {}'.format(it))
            print('Loss: {:.4}'.format(loss))
            print()

            numTest = len(x_test)
            batch_elements = [randint(0, numTest - 1) for n in range(0, mb_size)]
            t_X_mb = x_test[batch_elements, :]
            t_c_mb = c_test[batch_elements, :]
            t_X_mb = torch.from_numpy(t_X_mb).float().to(device)
            t_c_mb = torch.from_numpy(t_c_mb).float().to(device)
            c = t_c_mb
            #print(t_c_mb)

            gen_x, t_recon_obs = cvae.inference(t_c_mb[:, 2:obs_dim+2], t_c_mb[:, 0:2], mb_size)
            gen_x = gen_x.detach()
            t_X_mb = t_X_mb.detach()

            c = c.detach()
            X_mb = X_mb.detach()
            t_recon_obs = t_recon_obs.detach()
            print(t_recon_obs[0])
            print(t_c_mb[0, 2:obs_dim+2])

            show_distribution(c[0, 2:56*56+2].numpy(), c[0, 0:2].numpy(), X_mb.numpy(), t_recon_obs[0].numpy(), t_X_mb.numpy(), gen_x.numpy(), it, prefix)
        if it % 10000 == 0:
            if only_mlp == True:
                torch.save(cvae.state_dict(), os.path.join("th_models", prefix, "only_mlp-cvae-x{:d}-{:d}.pt".format(x_dim, it)))
            else:
                torch.save(cvae.state_dict(), os.path.join("th_models", prefix, "cvae-x{:d}-{:d}.pt".format(x_dim, it)))

data_file = "/home/lys/catkin_ws/src/crawler_crane/crane_tutorials/crane_planning_data/train_data1.pkl"
task_data, x_train, c_train, x_test, c_test = generate_data(3, data_file)
train(3, x_train, c_train, x_test, c_test, prefix = "goal", only_mlp=False)