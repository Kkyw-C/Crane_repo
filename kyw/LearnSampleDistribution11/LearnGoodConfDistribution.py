# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from random import randint
from planning_data_visual_tool import PlanningDataVisualTool

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def generate_data(x_dim, data_file):
    with open(data_file, "rb") as data_f:
        data = pickle.load(data_f)

    task_data = data['task_1']
    # print(task_data)
    # vt = PlanningDataVisualTool()
    # vt.visualize_global_map('gray')
    # vt.visualize_local_map(task_data['lift_obj_pose_init'], task_data['obs_init'], 'green')
    # vt.visualize_local_map(task_data['lift_obj_pose_goal'], task_data['obs_goal'], 'red')
    # plt.scatter(task_data['xytheta'][:, 0], task_data['xytheta'][:, 1], color='gray')

    # 加载x
    xytheta = task_data['xytheta']
    xytheta[:, 0] -= task_data['lift_obj_pose_init'][0]   # x
    xytheta[:, 1] -= task_data['lift_obj_pose_init'][1]   # y

    # 将样本x、y坐标转换为极角坐标
    xytheta[:, 0], xytheta[:, 1] = cart2pol(xytheta[:, 0], xytheta[:, 1])
    xytheta[:, 0] = (xytheta[:, 0] - 0.5 * (10 + 28)) / (28 - 0.5 * (10 + 28))   # 将r归一化到[-1, 1]

    if x_dim == 2:
        x = xytheta[:, 0:x_dim]
    else:
        x = xytheta
    n = len(x)

    # 加载被吊物目标位置和障碍物信息，作为条件c
    xytheta_goal = task_data['lift_obj_pose_goal']
    xytheta_goal[0] -= task_data['lift_obj_pose_init'][0]
    xytheta_goal[1] -= task_data['lift_obj_pose_init'][1]
    xytheta_goal = xytheta_goal[0:2]

    # vt.visualize_local_map(task_data['lift_obj_pose_init'] - task_data['lift_obj_pose_init'], task_data['obs_init'], 'green')
    # vt.visualize_local_map(task_data['lift_obj_pose_goal'], task_data['obs_goal'], 'red')
    # plt.scatter([0], [0], s = 5, c = 'green')
    # plt.scatter([xytheta_goal[0]], [xytheta_goal[1]], s = 5, c = 'red')
    # plt.scatter(x[:, 0], x[:, 1], c = 'green')
    # plt.show()

    obs = task_data['obs_init']
    c = np.concatenate([list(xytheta_goal), list(obs.flatten())], axis=0)
    c = np.repeat([c], n, axis=0)
    #print(c)

    # 将数据分为训练集和测试集
    # x = np.array(x)
    # c = np.array(c)
    n_train = int(n * 0.8)
    x_train = x[0:n_train, :]
    c_train = c[0:n_train, :]
    x_test = x[n_train:n, :]
    c_test = c[n_train:n, :]

    return task_data, x_train, c_train, x_test, c_test


def generate_data1(x_dim, data_file):
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

        obs = task_data['obs_init'] * 256  #
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

# def show_distribution(samples, recon_samples, gen_samples, obs, it, prefix):
#     #print(samples.shape, recon_samples.shape)
#     scene_size = 250.0
#     #plt.figure(figsize=(scene_size, scene_size))
#     plt.axes().set_aspect('equal')  # 为了绘制的图形不变形
#     plt.title('Lifting Scene')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.xlim([-0.5*scene_size, 0.5*scene_size])  # 设置x轴的边界
#     plt.ylim([-0.5*scene_size, 0.5*scene_size])  # 设置y轴的边界
#
#     rows = cols = int(len(obs) ** 0.5)
#     grid = np.reshape(obs, (rows, cols))
#     x, y = np.nonzero(grid)
#     x, y = x*scene_size/rows - 0.5*scene_size, y*scene_size/rows - 0.5*scene_size
#     plt.scatter(x, y, color='gray', s=10, alpha=0.8)
#
#     # if len(samples.shape) == 1:
#     #     samples = np.reshape(samples,(1, -1))
#     # plt.scatter(samples[:, 0], samples[:, 1], color='green', s=7, alpha=0.3)
#     #
#     # if len(recon_samples.shape) == 1:
#     #     recon_samples = np.reshape(recon_samples,(1, -1))
#     # plt.scatter(recon_samples[:, 0], recon_samples[:, 1], color='blue', s=7, alpha=0.3)
#
#     if len(gen_samples.shape) == 1:
#         gen_samples = np.reshape(gen_samples,(1, -1))
#     plt.scatter(gen_samples[:, 0], gen_samples[:, 1], color='red', s=7, alpha=0.3)
#
#     sample_dim = samples.shape[1]
#     plt.savefig("figs/{:s}/fig-x{:d}-{:d}.png".format(prefix, sample_dim, it), dpi=300)
#
#     #plt.show()
#
#     plt.clf()
#     plt.close()

vt = PlanningDataVisualTool()
def show_distribution(obs, goal, origional_samples, recon_samples, test_samples, gen_samples, it, prefix):
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

    # 绘制原始样本
    origional_samples[:, 0] = origional_samples[:, 0] * (28 - 0.5 * (10 + 28)) + 0.5 * (10 + 28)   # 将归一化后的值恢复回来
    origional_samples[:, 0], origional_samples[:, 1] = pol2cart(origional_samples[:, 0], origional_samples[:, 1])   # 极坐标转换为笛卡尔坐标
    #print(origional_samples)
    plt.scatter(origional_samples[:, 0], origional_samples[:, 1], s = 2, c = 'gray')

    # 绘制重构样本
    recon_samples[:, 0] = recon_samples[:, 0] * (28 - 0.5 * (10 + 28)) + 0.5 * (10 + 28)  # 将归一化后的值恢复回来
    recon_samples[:, 0], recon_samples[:, 1] = pol2cart(recon_samples[:, 0], recon_samples[:, 1])  # 极坐标转换为笛卡尔坐标
    plt.scatter(recon_samples[:, 0], recon_samples[:, 1], s=2, c='green')

    # 绘制生成样本
    test_samples[:, 0] = test_samples[:, 0] * (28 - 0.5 * (10 + 28)) + 0.5 * (10 + 28)  # 将归一化后的值恢复回来
    test_samples[:, 0], test_samples[:, 1] = pol2cart(test_samples[:, 0], test_samples[:, 1])  # 极坐标转换为笛卡尔坐标
    plt.scatter(test_samples[:, 0], test_samples[:, 1], s=2, c='yellow')

    # 绘制生成样本
    gen_samples[:, 0] = gen_samples[:, 0] * (28 - 0.5 * (10 + 28)) + 0.5 * (10 + 28)  # 将归一化后的值恢复回来
    gen_samples[:, 0], gen_samples[:, 1] = pol2cart(gen_samples[:, 0], gen_samples[:, 1])  # 极坐标转换为笛卡尔坐标
    plt.scatter(gen_samples[:, 0], gen_samples[:, 1], s=2, c='blue')




    # plt.scatter(samples[:, 0], samples[:, 1], color='green', s=1, alpha=0.3)
    # # #
    # # # if len(recon_samples.shape) == 1:
    # # #     recon_samples = np.reshape(recon_samples,(1, -1))
    # # # plt.scatter(recon_samples[:, 0], recon_samples[:, 1], color='blue', s=7, alpha=0.3)
    #
    # if len(gen_samples.shape) == 1:
    #     gen_samples = np.reshape(gen_samples,(1, -1))
    # plt.scatter(gen_samples[:, 0], gen_samples[:, 1], color='blue', s=1, alpha=0.3)
    #
    # mean = np.mean(samples, axis=0)
    # # print(mean)
    # plt.scatter(mean[0], mean[1], color='black', s=30, alpha=0.8)

    sample_dim = gen_samples.shape[1]
    plt.savefig("figs/{:s}/fig-x{:d}-{:d}.png".format(prefix, sample_dim, it), dpi=300)

    #plt.show()

    plt.clf()
    plt.close()



class VAE(nn.Module):

    def __init__(self, x_dim, latent_size, conditional=False, c_dim=0):

        super().__init__()

        self.conditional = conditional

        if self.conditional:
            assert c_dim > 0

        assert type(x_dim) == int
        assert type(latent_size) == int

        self.latent_size = latent_size

        self.encoder = Encoder(x_dim, latent_size, conditional, c_dim)
        self.decoder = Decoder(x_dim, latent_size, conditional, c_dim)

    def forward(self, x, c=None):
        batch_size = x.size(0)

        if self.conditional:
            means, log_var = self.encoder(x, c)

            std = torch.exp(0.9 * log_var)
            eps = torch.randn([batch_size, self.latent_size])
            z = eps * std + means

            recon_x = self.decoder(z, c)

            return recon_x, means, log_var, z
        else:
            means, log_var = self.encoder(x, c=None)
            std = torch.exp(0.5 * log_var)
            eps = torch.randn([batch_size, self.latent_size])
            z = eps * std + means
            recon_x = self.decoder(z, c=None)

            return recon_x, means, log_var, z


    def inference(self, n=1, c=None):

        batch_size = n
        z = torch.rand([batch_size, self.latent_size])
        z = z * 5

        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):

    def __init__(self, x_dim, latent_size, conditional, condition_dim):

        super().__init__()

        self.conditional = conditional
        input_size = x_dim
        if self.conditional:
            input_size = x_dim + condition_dim

        self.MLP = nn.Sequential()

        self.MLP.add_module('L0', nn.Linear(input_size, 512))
        self.MLP.add_module('A0', nn.ReLU())
        #self.MLP.add_module('D0', nn.Dropout(0.5))
        self.MLP.add_module('L1', nn.Linear(512, 512))
        self.MLP.add_module('A1', nn.ReLU())

        self.linear_means = nn.Linear(512, latent_size)
        self.linear_log_var = nn.Linear(512, latent_size)

    def forward(self, x, c=None):

        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, x_dim, latent_size, conditional, condition_dim):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + condition_dim
        else:
            input_size = latent_size

        self.MLP.add_module('L0', nn.Linear(input_size, 512))
        self.MLP.add_module('A0', nn.ReLU())
        #self.MLP.add_module('D0', nn.Dropout(0.5))
        self.MLP.add_module('L1', nn.Linear(512, 512))
        self.MLP.add_module('A1', nn.ReLU())
        self.MLP.add_module('O', nn.Linear(512, x_dim))

    def forward(self, z, c):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x


def test(x_test, c_test):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x_dim = 3
    vae = VAE(
        x_dim=x_dim,  # conf=(7)
        latent_size=3,
        conditional=True,
        c_dim=56 * 56 + 2  # obs=(84*84), init=(3), goal=(3)
    ).to(device)

    print(vae)

    vae.load_state_dict(torch.load('th_models/init/vae-x3-70000.pt'))

    obs_dict = np.load("local_bin_map_dict_56x56.npy")
    obs_dict = obs_dict.item()

    for (key, value) in obs_dict.items():
        key = key[1:len(key)-1]
        lift_obj_pos = key.split(',')
        lift_obj_pos = [ float(x) for x in lift_obj_pos ]
        #print(value)

        num = 1000
        c = np.concatenate((lift_obj_pos, value.flatten()))
        c = np.repeat([c], num, axis=0)
        c = torch.from_numpy(c).float()
        gen_x = vae.inference(num, c)
        gen_x = gen_x.detach().numpy()
        show_distribution("local_bin_map_dict_56x56.npy", lift_obj_pos, gen_x, 0, "init")

def train(x_dim, x_train, c_train, x_test, c_test, prefix):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def loss_fn(recon_x, x, mean, log_var):

        w = np.array([[1, 1, 1, 0.5, 0.5, 0.5]])
        w = torch.from_numpy(w).float()
        BCE = torch.nn.functional.mse_loss(
            recon_x.view(-1, x_dim), x.view(-1, x_dim), reduction='mean')
        KLD = 10 ** -6 * 2 * torch.mean(log_var.exp() + mean.pow(2) - 1. - log_var, 1)

        vae_loss = torch.mean(BCE + KLD)

        return vae_loss
    #
    # vae:
    # 可以考虑loss加权重
    # 推理时现在是均匀采样
    # z = torch.rand([batch_size, self.latent_size])
    # z = z * 5


可以考虑高斯分布


    vae = VAE(
        x_dim=x_dim,                 # conf=(7)
        latent_size=3,
        conditional=True,
        c_dim=56*56+2          # obs=(84*84), init=(2), goal=(2)
    ).to(device)

    print(vae)

    optimizer = torch.optim.Adam(vae.parameters(), lr=10e-5)

    it = 0
    numTrain = len(x_train)
    mb_size = 128
    for it in range(it, it + 500001):
        # randomly generate batches
        batch_elements = [randint(0, numTrain - 1) for n in range(0, mb_size)]
        X_mb = x_train[batch_elements, :]
        c_mb = c_train[batch_elements, :]
        X_mb = torch.from_numpy(X_mb).float().to(device)
        c_mb = torch.from_numpy(c_mb).float().to(device)
        #print(X_mb.size(), c_mb.size())

        if vae.conditional:
            recon_x, mean, log_var, z = vae(X_mb, c_mb)
        else:
            recon_x, mean, log_var, z = vae(X_mb, c=None)

        loss = loss_fn(recon_x, X_mb, mean, log_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 500 == 0:
            print('Iter: {}'.format(it))
            print('Loss: {:.4}'.format(loss))
            print()

            #num = 100
            # c = c_test[0, :]
            # c = np.repeat([c], num, axis=0)
            # c = torch.from_numpy(c).float()
            numTest = len(x_test)
            batch_elements = [randint(0, numTest - 1) for n in range(0, mb_size)]
            t_X_mb = x_test[batch_elements, :]
            t_c_mb = c_test[batch_elements, :]
            t_X_mb = torch.from_numpy(t_X_mb).float().to(device)
            t_c_mb = torch.from_numpy(t_c_mb).float().to(device)
            c = t_c_mb

            gen_x = vae.inference(mb_size, c)
            gen_x = gen_x.detach()
            t_X_mb = t_X_mb.detach()

            c = c.detach()
            X_mb = X_mb.detach()
            recon_x = recon_x.detach()

            show_distribution(c[0, 2:56*56+2].numpy(), c[0, 0:2].numpy(), X_mb.numpy(), recon_x.numpy(), t_X_mb.numpy(), gen_x.numpy(), it, prefix)
        if it % 10000 == 0:
            torch.save(vae.state_dict(), os.path.join("th_models", prefix, "vae-x{:d}-{:d}.pt".format(x_dim, it)))

data_file = "/home/lys/catkin_ws/src/crawler_crane/crane_tutorials/crane_planning_data/train_data.pkl"
task_data, x_train, c_train, x_test, c_test = generate_data1(3, data_file)
train(3, x_train, c_train, x_test, c_test, prefix = "init")

#test(x_test, c_test)


