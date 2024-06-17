import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import os
import csv
from random import randint, random
import time

import torch
import torch.nn as nn


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

            std = torch.exp(0.5 * log_var)
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
        z = torch.randn([batch_size, self.latent_size])

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

# neural network parameters
mb_size = 256
h_Q_dim = 512
h_P_dim = 512
X_dim = 6
c_dim = 133
z_dim = 3
lr = 1e-4


# change conditions to occupancy grid
def isSampleFree(sample, obs, dimW):
    for o in range(0, int(obs.shape[0] / (2 * dimW))):
        isFree = 0
        for d in range(0, sample.shape[0]):
            if (sample[d] < obs[2 * dimW * o + d] or sample[d] > obs[2 * dimW * o + d + dimW]):
                isFree = 1
                break
        if isFree == 0:
            return 0
    return 1


def generate_data(file_name):
    f = open(file_name, 'r')
    reader = csv.reader(f, delimiter=',')

    # problem dimensions
    dim = 6
    dataElements = dim + 3 * 3 + 2 * dim  # sample (6D), gap1 (2D, 1D orientation), gap2, gap3, init (6D), goal (6D)
    count = 0
    data_list = []
    for row in reader:
        data_list.append(row[0:dataElements])

    data = np.array(data_list, dtype='d')
    numEntries = data.shape[0]

    # split the inputs and conditions into test train (to be processed in the next step into an occupancy grid representation)
    ratioTestTrain = 0.8;
    numTrain = int(numEntries * ratioTestTrain)

    X_train = data[0:numTrain, 0:dim]  # state: x, y, z, xdot, ydot, zdot
    c_train = data[0:numTrain, dim:dataElements]  # conditions: gaps, init (6), goal (6)

    X_test = data[numTrain:numEntries, 0:dim]
    c_test = data[numTrain:numEntries, dim:dataElements]
    numTest = X_test.shape[0]



    gridSize = 11
    dimW = 3
    plotOn = False;

    # process data into occupancy grid
    conditions = data[0:numEntries, dim:dataElements]
    conditionsOcc = np.zeros([numEntries, gridSize * gridSize])
    occGridSamples = np.zeros([gridSize * gridSize, 2])
    gridPointsRange = np.linspace(0, 1, num=gridSize)

    idx = 0;
    for i in gridPointsRange:
        for j in gridPointsRange:
            occGridSamples[idx, 0] = i
            occGridSamples[idx, 1] = j
            idx += 1;

    start = time.time();
    for j in range(0, numEntries, 1):
        dw = 0.1
        dimW = 3
        gap1 = conditions[j, 0:3]
        gap2 = conditions[j, 3:6]
        gap3 = conditions[j, 6:9]
        init = conditions[j, 9:15]
        goal = conditions[j, 15:21]

        obs1 = [0, gap1[1] - dw, -0.5, gap1[0], gap1[1], 1.5]
        obs2 = [gap2[0] - dw, 0, -0.5, gap2[0], gap2[1], 1.5];
        obs3 = [gap2[0] - dw, gap2[1] + dw, -0.5, gap2[0], 1, 1.5];
        obs4 = [gap1[0] + dw, gap1[1] - dw, -0.5, gap3[0], gap1[1], 1.5];
        obs5 = [gap3[0] + dw, gap1[1] - dw, -0.5, 1, gap1[1], 1.5];
        obs = np.concatenate((obs1, obs2, obs3, obs4, obs5), axis=0)

        if j % 5000 == 0:
            print('Iter: {}'.format(j))

        occGrid = np.zeros(gridSize * gridSize)
        for i in range(0, gridSize * gridSize):
            occGrid[i] = isSampleFree(occGridSamples[i, :], obs)
        conditionsOcc[j, :] = occGrid

        if plotOn:
            fig1 = plt.figure(figsize=(10, 6), dpi=80)
            ax1 = fig1.add_subplot(111, aspect='equal')
            for i in range(0, obs.shape[0] / (2 * dimW)):  # plot obstacle patches
                ax1.add_patch(
                    patches.Rectangle(
                        (obs[i * 2 * dimW], obs[i * 2 * dimW + 1]),  # (x,y)
                        obs[i * 2 * dimW + dimW] - obs[i * 2 * dimW],  # width
                        obs[i * 2 * dimW + dimW + 1] - obs[i * 2 * dimW + 1],  # height
                        alpha=0.6
                    ))
            for i in range(0, gridSize * gridSize):  # plot occupancy grid
                if occGrid[i] == 0:
                    plt.scatter(occGridSamples[i, 0], occGridSamples[i, 1], color="red", s=70, alpha=0.8)
                else:
                    plt.scatter(occGridSamples[i, 0], occGridSamples[i, 1], color="green", s=70, alpha=0.8)
            plt.show()
    end = time.time();
    print('Time: ', end - start)

    cs = np.concatenate((data[0:numEntries, dim + 3 * dimW:dataElements], conditionsOcc), axis=1)  # occ, init, goal
    c_dim = cs.shape[1]
    c_gapsInitGoal = c_test
    c_train = cs[0:numTrain, :]
    c_test = cs[numTrain:numEntries, :]

    np.save("X_train_file", X_train)
    np.save("c_train_file", c_train)
    np.save("c_test_file", c_test)
    np.save("c_gapsInitGoal_file", c_gapsInitGoal)
    np.save("occGridSamples_file", occGridSamples)




def test(vae, c_test, c_gapsInitGoal, occGridSamples):
    dim = 6
    gridSize = 11
    # plot the latent space
    num_viz = 3000
    numTest = len(c_test)

    vizIdx = randint(0, numTest - 1)
    c_sample_seed = c_test[vizIdx, :]
    c_sample = np.repeat([c_sample_seed], num_viz, axis=0)
    c_viz = c_gapsInitGoal[vizIdx, :]
    c_sample = torch.from_numpy(c_sample).float().to("cpu")

    if vae.conditional:
        y_viz = vae.inference(n=num_viz, c=c_sample)
    else:
        y_viz = vae.inference(n=num_viz)

    y_viz = y_viz.detach().numpy()

    fig1 = plt.figure(figsize=(10, 6), dpi=80)
    ax1 = fig1.add_subplot(111, aspect='equal')

    plt.scatter(y_viz[:, 0], y_viz[:, 1], color="green", s=70, alpha=0.1)

    dw = 0.1
    dimW = 3
    gap1 = c_viz[0:3]
    gap2 = c_viz[3:6]
    gap3 = c_viz[6:9]
    init = c_viz[9:15]
    goal = c_viz[15:21]

    obs1 = [0, gap1[1] - dw, -0.5, gap1[0], gap1[1], 1.5]
    obs2 = [gap2[0] - dw, 0, -0.5, gap2[0], gap2[1], 1.5];
    obs3 = [gap2[0] - dw, gap2[1] + dw, -0.5, gap2[0], 1, 1.5];
    obs4 = [gap1[0] + dw, gap1[1] - dw, -0.5, gap3[0], gap1[1], 1.5];
    obs5 = [gap3[0] + dw, gap1[1] - dw, -0.5, 1, gap1[1], 1.5];
    obsBounds = [-0.1, -0.1, -0.5, 0, 1.1, 1.5,
                 -0.1, -0.1, -0.5, 1.1, 0, 1.5,
                 -0.1, 1, -0.5, 1.1, 1.1, 1.5,
                 1, -0.1, -0.5, 1.1, 1.1, 1.5, ]

    obs = np.concatenate((obs1, obs2, obs3, obs4, obs5, obsBounds), axis=0)
    for i in range(0, int(obs.shape[0] / (2 * dimW))):
        ax1.add_patch(
            patches.Rectangle(
                (obs[i * 2 * dimW], obs[i * 2 * dimW + 1]),  # (x,y)
                obs[i * 2 * dimW + dimW] - obs[i * 2 * dimW],  # width
                obs[i * 2 * dimW + dimW + 1] - obs[i * 2 * dimW + 1],  # height
                alpha=0.6
            ))

    for i in range(0, gridSize * gridSize):  # plot occupancy grid
        cIdx = i + 2 * dim
        if c_sample_seed[cIdx] == 0:
            plt.scatter(occGridSamples[i, 0], occGridSamples[i, 1], color="red", s=50, alpha=0.7)
        else:
            plt.scatter(occGridSamples[i, 0], occGridSamples[i, 1], color="green", s=50, alpha=0.7)

    plt.scatter(init[0], init[1], color="red", s=250, edgecolors='black')  # init
    plt.scatter(goal[0], goal[1], color="blue", s=250, edgecolors='black')  # goal

    plt.show()

    plt.figure(figsize=(10, 6), dpi=80)
    viz1 = 1;
    viz2 = 4;
    plt.scatter(y_viz[:, viz1], y_viz[:, viz2], color="green", s=70, alpha=0.1)
    plt.scatter(c_viz[viz1 + 9], c_viz[viz2 + 9], color="red", s=250, edgecolors='black')  # init

    plt.scatter(c_viz[viz1 + 9 + dim], c_viz[viz2 + 9 + dim], color="blue", s=500, edgecolors='black')  # goal
    plt.show()



X_train = np.load("X_train.npy")
c_train = np.load("c_train.npy")
c_test = np.load("c_test.npy")
c_gapsInitGoal = np.load("c_gapsInitGoal.npy")
occGridSamples = np.load("occGridSamples.npy")
numTrain = X_train.shape[0]
X_dim = X_train.shape[1]
c_dim = c_train.shape[1]

def train():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def loss_fn(recon_x, x, mean, log_var):
        # print(recon_x.size())
        # print(x.size())

        w = np.array([[1, 1, 1, 0.5, 0.5, 0.5]])
        w = torch.from_numpy(w).float()
        BCE = torch.nn.functional.mse_loss(
            recon_x.view(-1, 3), x.view(-1, 3), reduction='mean')
        KLD = 10 ** -4 * 2 * torch.mean(log_var.exp() + mean.pow(2) - 1. - log_var, 1)

        vae_loss = torch.mean(BCE + KLD)

        return vae_loss

    vae = VAE(
        x_dim=6,
        latent_size=3,
        conditional=True,
        c_dim=133).to(device)

    print(vae)

    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    it = 0;


    for it in range(it, it + 500001):
        # randomly generate batches
        batch_elements = [randint(0, numTrain - 1) for n in range(0, mb_size)]
        X_mb = X_train[batch_elements, :]
        c_mb = c_train[batch_elements, :]
        X_mb = torch.from_numpy(X_mb).float().to(device)
        c_mb = torch.from_numpy(c_mb).float().to(device)

        #_, loss = sess.run([train_step, cvae_loss], feed_dict={X: X_mb, c: c_mb})
        if vae.conditional:
            recon_x, mean, log_var, z = vae(X_mb, c_mb)
        else:
            recon_x, mean, log_var, z = vae(X_mb, c=None)

        loss = loss_fn(recon_x, X_mb, mean, log_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 1000 == 0:
            print('Iter: {}'.format(it))
            print('Loss: {:.4}'.format(loss))
            print()

        if it % 10000 == 0:
            torch.save(vae.state_dict(), os.path.join("th_models", "vae-{:d}.pt".format(it)))
            test(vae, c_test, c_gapsInitGoal, occGridSamples)

train()