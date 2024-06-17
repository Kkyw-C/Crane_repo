import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from random import randint




# def generate_data(x_dim, init, confs_file, obs_file, init_lift_pose, goal_lift_pose):
#     confs = np.loadtxt(confs_file, delimiter=',', skiprows=1)  # 跳过第一行的表头
#     obs = np.loadtxt(obs_file)
#
#     num = len(confs)
#
#     if init == True:
#         x = confs[:, 0:x_dim]
#     else:
#         x = confs[:, 7:7+x_dim]
#
#     obs = obs.flatten()
#     c = np.concatenate((init_lift_pose, goal_lift_pose, obs))
#
#     c = np.repeat([c], num, axis=0)
#
#     num_train = int(num * 0.8)
#     x_train = x[0:num_train, :]
#     c_train = c[0:num_train, :]
#
#     x_test = x[num_train:num, :]
#     c_test = c[num_train:num, :]
#
#     return x_train, c_train, x_test, c_test

def generate_data(x_dim, confs_file, obs_file):
    confs = np.loadtxt(confs_file, delimiter=',')  # 跳过第一行的表头
    obs_dict = np.load(obs_file)
    obs_dict = obs_dict.item()

    num = len(confs)
    data = []
    for i in range(num):
        key = "({:d},{:d},{:d})".format(int(confs[i][7]), int(confs[i][8]), int(confs[i][9]))
        obs = obs_dict[key]
        obs = obs.flatten()
        # # 将样本的绝对坐标转换为相对坐标
        # confs[i][0] -= confs[i][7]
        # confs[i][1] -= confs[i][8]
        # confs[i][7] -= confs[i][7]
        # confs[i][8] -= confs[i][8]
        data.append(np.concatenate((confs[i], obs)))

    data = np.array(data)
    np.random.shuffle(data)
    #print(data)
    num_train = int(num * 0.8)
    data_dim = len(data[0])
    x_train = data[0:num_train, 0:x_dim]
    c_train = data[0:num_train, 7:data_dim]
    #print(len(c_train[0]))

    x_test = data[num_train:num, 0:x_dim]
    c_test = data[num_train:num, 7:data_dim]

    #print(data[0])
    # print(x_train[0])
    # print(c_train[0])

    return x_train, c_train, x_test, c_test

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

def show_distribution(obs_file, lift_obj_pos, gen_samples, it, prefix):   # gen_samples, obs, obs_file, it, prefix
    scene_size = 250.0
    plt.axes().set_aspect('equal')  # 为了绘制的图形不变形
    plt.title('Lifting Scene')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim([-0.5*scene_size, 0.5*scene_size])  # 设置x轴的边界
    plt.ylim([-0.5*scene_size, 0.5*scene_size])  # 设置y轴的边界

    # 绘制全局地图
    global_map = np.loadtxt("bin_map.txt")
    rows = cols = int(len(global_map))
    x, y = np.nonzero(global_map)
    x, y = x*scene_size/rows - 0.5*scene_size, y*scene_size/rows - 0.5*scene_size
    plt.scatter(x, y, color='gray', s=1, alpha=0.3)

    obs_dict = np.load(obs_file)
    obs_dict = obs_dict.item()
    key = "({:d},{:d},{:d})".format(int(lift_obj_pos[0]), int(lift_obj_pos[1]), int(lift_obj_pos[2]))
    local_map = obs_dict[key]
    x, y = np.nonzero(local_map)
    x = x - 0.5 * len(local_map) + lift_obj_pos[0]
    y = y - 0.5 * len(local_map) + lift_obj_pos[1]
    #print(local_map)

    plt.scatter(x, y, color='yellow', s=1, alpha=0.3)
    plt.scatter(lift_obj_pos[0], lift_obj_pos[1], color='red', s=20)

    confs = np.loadtxt("collision_free_confs.txt", delimiter=',')
    num_confs = len(confs)
    #print(num_confs)
    samples = []
    n = 0
    for i in range(num_confs):
        str = "({:d},{:d},{:d})".format(int(confs[i][7]), int(confs[i][8]), int(confs[i][9]))
        if str == key:
            samples.append([confs[i][0], confs[i][1]])
            n += 1
            if n == 2000:
                break

    #print(samples)
    samples = np.array(samples)


    plt.scatter(samples[:, 0], samples[:, 1], color='green', s=1, alpha=0.3)
    # #
    # # if len(recon_samples.shape) == 1:
    # #     recon_samples = np.reshape(recon_samples,(1, -1))
    # # plt.scatter(recon_samples[:, 0], recon_samples[:, 1], color='blue', s=7, alpha=0.3)

    if len(gen_samples.shape) == 1:
        gen_samples = np.reshape(gen_samples,(1, -1))
    plt.scatter(gen_samples[:, 0], gen_samples[:, 1], color='blue', s=1, alpha=0.3)

    mean = np.mean(samples, axis=0)
    # print(mean)
    plt.scatter(mean[0], mean[1], color='black', s=30, alpha=0.8)

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

        self.MLP.add_module('L0', nn.Linear(input_size, 512, bias=False))
        self.MLP.add_module('A0', nn.ReLU())
        self.MLP.add_module('D0', nn.Dropout(0.5))
        self.MLP.add_module('L1', nn.Linear(512, 512, bias=False))
        self.MLP.add_module('A1', nn.ReLU())

        self.linear_means = nn.Linear(512, latent_size, bias=False)
        self.linear_log_var = nn.Linear(512, latent_size, bias=False)

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

        self.MLP.add_module('L0', nn.Linear(input_size, 512, bias=False))
        self.MLP.add_module('A0', nn.ReLU())
        self.MLP.add_module('D0', nn.Dropout(0.5))
        self.MLP.add_module('L1', nn.Linear(512, 512, bias=False))
        self.MLP.add_module('A1', nn.ReLU())
        self.MLP.add_module('O', nn.Linear(512, x_dim, bias=False))

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
        latent_size=2,
        conditional=True,
        c_dim=56 * 56 + 3  # obs=(84*84), init=(3), goal=(3)
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
        KLD = 10 ** -2 * 2 * torch.mean(log_var.exp() + mean.pow(2) - 1. - log_var, 1)

        vae_loss = torch.mean(BCE + KLD)

        return vae_loss

    vae = VAE(
        x_dim=x_dim,                 # conf=(7)
        latent_size=2,
        conditional=True,
        c_dim=56*56+3          # obs=(84*84), init=(3), goal=(3)
    ).to(device)

    print(vae)

    optimizer = torch.optim.Adam(vae.parameters(), lr=10e-5)

    it = 0
    numTrain = len(x_train)
    mb_size = 64
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

            num = 100
            c = c_test[0, :]
            c = np.repeat([c], num, axis=0)
            c = torch.from_numpy(c).float()
            gen_x = vae.inference(num, c)
            gen_x = gen_x.detach().numpy()
            #show_distribution(X_mb.numpy(), recon_x.detach().numpy(), gen_x, c[0, 3:int(28*28+3)].numpy(), it, prefix)
            show_distribution("local_bin_map_dict_56x56.npy", c[0, 0:3].numpy(), gen_x, it, prefix)
        if it % 10000 == 0:
            torch.save(vae.state_dict(), os.path.join("th_models", prefix, "vae-x{:d}-{:d}.pt".format(x_dim, it)))

# x_train, c_train, x_test, c_test = generate_data(x_dim = 3, init=True, confs_file="RRTCon.csv", obs_file="bin_map_84x84.txt", init_lift_pose=[8.0, 110.0, 10], goal_lift_pose=[-18.0, 8.0, 12.0])
# train(3, x_train, c_train, x_test, c_test, prefix = "init")
#
# x_train, c_train, x_test, c_test = generate_data(x_dim = 3, init=False, confs_file="RRTCon.csv", obs_file="bin_map_84x84.txt", init_lift_pose=[8.0, 110.0, 10], goal_lift_pose=[-18.0, 8.0, 12.0])
# train(3, x_train, c_train, x_test, c_test, prefix = "goal")
#
#
# x_train, c_train, x_test, c_test = generate_data(x_dim = 7, init=True, confs_file="RRTCon.csv", obs_file="bin_map_84x84.txt", init_lift_pose=[8.0, 110.0, 10], goal_lift_pose=[-18.0, 8.0, 12.0])
# train(7, x_train, c_train, x_test, c_test, prefix = "init")
#
# x_train, c_train, x_test, c_test = generate_data(x_dim = 7, init=False, confs_file="RRTCon.csv", obs_file="bin_map_84x84.txt", init_lift_pose=[8.0, 110.0, 10], goal_lift_pose=[-18.0, 8.0, 12.0])
# train(7, x_train, c_train, x_test, c_test, prefix = "goal")

#
x_train, c_train, x_test, c_test = generate_data(x_dim = 3,confs_file="collision_free_confs.txt", obs_file="local_bin_map_dict_56x56.npy")
train(3, x_train, c_train, x_test, c_test, prefix = "init")

#test(x_test, c_test)


