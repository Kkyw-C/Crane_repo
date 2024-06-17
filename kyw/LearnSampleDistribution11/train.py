import os
import time
import torch
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from collections import defaultdict
import numpy as np
import csv
import matplotlib.patches as patches
from random import randint

from models import VAE

# neural network parameters
mb_size = 256
h_Q_dim = 512
h_P_dim = 512

c = 0
lr = 1e-4

# problem dimensions
dim = 6
dataElements = dim + 3 * 3 + 2 * dim  # sample (6D), gap1 (2D, 1D orientation), gap2, gap3, init (6D), goal (6D)

z_dim = 3  # latent
X_dim = dim  # samples
y_dim = dim  # reconstruction of the original point
c_dim = dataElements - dim  # dimension of conditioning variable

# read in data from csv
filename = 'narrowDataFile(orginal).txt'
f = open(filename, 'r')
reader = csv.reader(f, delimiter=',')

count = 0
data_list = []
for row in reader:
    data_list.append(list(row[0:dataElements]))

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

# # change conditions to occupancy grid
# def isSampleFree(sample, obs):
#     n = int(obs.shape[0] / (2 * dimW))
#     for o in range(0, n):
#         isFree = 0
#         for d in range(0, sample.shape[0]):
#             if (sample[d] < obs[2 * dimW * o + d] or sample[d] > obs[2 * dimW * o + d + dimW]):
#                 isFree = 1
#                 break
#         if isFree == 0:
#             return 0
#     return 1
#
#
#
#
# # process data into occupancy grid
# conditions = data[0:numEntries, dim:dataElements]
# conditionsOcc = np.zeros([numEntries, gridSize * gridSize])
# occGridSamples = np.zeros([gridSize * gridSize, 2])
# gridPointsRange = np.linspace(0, 1, num=gridSize)
#
# idx = 0;
# for i in gridPointsRange:
#     for j in gridPointsRange:
#         occGridSamples[idx, 0] = i
#         occGridSamples[idx, 1] = j
#         idx += 1;
#
#
# start = time.time();
# fig1 = plt.figure(figsize=(10, 6), dpi=80)
# ax1 = fig1.add_subplot(111, aspect='equal')
# plt.ion()  # 打开交互式绘图interactive
# for j in range(0, numEntries, 1):
#     dw = 0.1
#     dimW = 3
#     gap1 = conditions[j, 0:3]
#     gap2 = conditions[j, 3:6]
#     gap3 = conditions[j, 6:9]
#     init = conditions[j, 9:15]
#     goal = conditions[j, 15:21]
#
#     obs1 = [0, gap1[1] - dw, -0.5, gap1[0], gap1[1], 1.5]
#     obs2 = [gap2[0] - dw, 0, -0.5, gap2[0], gap2[1], 1.5];
#     obs3 = [gap2[0] - dw, gap2[1] + dw, -0.5, gap2[0], 1, 1.5];
#     obs4 = [gap1[0] + dw, gap1[1] - dw, -0.5, gap3[0], gap1[1], 1.5];
#     obs5 = [gap3[0] + dw, gap1[1] - dw, -0.5, 1, gap1[1], 1.5];
#     obs = np.concatenate((obs1, obs2, obs3, obs4, obs5), axis=0)
#
#     if j % 5000 == 0:
#         print('Iter: {}'.format(j))
#
#     occGrid = np.zeros(gridSize * gridSize)
#     for i in range(0, gridSize * gridSize):
#         occGrid[i] = isSampleFree(occGridSamples[i, :], obs)
#     conditionsOcc[j, :] = occGrid
#
#     if plotOn:
#         plt.cla()
#         for i in range(0, int(obs.shape[0] / (2 * dimW))):  # plot obstacle patches
#             ax1.add_patch(
#                 patches.Rectangle(
#                     (obs[i * 2 * dimW], obs[i * 2 * dimW + 1]),  # (x,y)
#                     obs[i * 2 * dimW + dimW] - obs[i * 2 * dimW],  # width
#                     obs[i * 2 * dimW + dimW + 1] - obs[i * 2 * dimW + 1],  # height
#                     alpha=0.6
#                 ))
#         for i in range(0, gridSize * gridSize):  # plot occupancy grid
#             if occGrid[i] == 0:
#                 plt.scatter(occGridSamples[i, 0], occGridSamples[i, 1], color="red", s=70, alpha=0.8)
#             else:
#                 plt.scatter(occGridSamples[i, 0], occGridSamples[i, 1], color="green", s=70, alpha=0.8)
#         plt.pause(0.01)
#         #plt.ioff()  # 关闭交互式绘图
#         #plt.show(block=False)
#
# end = time.time();
# print('Time: ', end - start)
#
# cs = np.concatenate((data[0:numEntries, dim + 3 * dimW:dataElements], conditionsOcc), axis=1)  # occ, init, goal
# c_dim = cs.shape[1]
# c_gapsInitGoal = c_test
# c_train = cs[0:numTrain, :]
# c_test = cs[numTrain:numEntries, :]

# np.save("X_train", X_train)
# np.save("c_train", c_train)
# np.save("c_test", c_test)

X_train = np.load("X_train.npy")
c_train = np.load("c_train.npy")
c_test = np.load("c_test.npy")
X_dim = X_train.shape[1]
c_dim = c_train.shape[1]

# print(X_train.shape)
# print(c_train.shape)
# print(c_test.shape)

def main(args):

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ts = time.time()

    def loss_fn(recon_x, x, mean, log_var):
        # print(recon_x.size())
        # print(x.size())

        w = np.array([[1, 1, 1, 0.5, 0.5, 0.5]])
        w = torch.from_numpy(w).float()
        BCE = torch.nn.functional.mse_loss(
            recon_x.view(-1, X_dim), x.view(-1, X_dim), reduction='mean')
        KLD = 10 ** -4 * 2 * torch.mean(log_var.exp() + mean.pow(2) - 1. - log_var, 1)

        vae_loss = torch.mean(BCE+KLD)

        return vae_loss

    vae = VAE(
        encoder_layer_sizes=args.encoder_layer_sizes,
        latent_size=args.latent_size,
        decoder_layer_sizes=args.decoder_layer_sizes,
        conditional=args.conditional,
        condition_dim=c_dim if args.conditional else 0).to(device)

    print(vae)

    optimizer = torch.optim.Adam(vae.parameters(), lr=args.learning_rate)

    logs = defaultdict(list)

    for epoch in range(args.epochs):

        iteration = 0;

        for iteration in range(iteration, iteration + 500001):
            # randomly generate batches
            batch_elements = [randint(0, numTrain - 1) for n in range(0, mb_size)]
            x = X_train[batch_elements, :]
            c = c_train[batch_elements, :]

            x = torch.from_numpy(x).float()
            c = torch.from_numpy(c).float()

            x, c = x.to(device), c.to(device)

            if args.conditional:
                recon_x, mean, log_var, z = vae(x, c)
            else:
                recon_x, mean, log_var, z = vae(x)

            loss = loss_fn(recon_x, x, mean, log_var)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logs['loss'].append(loss.item())

            if iteration % args.print_every == 0:
                print("Epoch {:02d}/{:02d} Batch {:04d}, Loss {:9.4f}".format(
                    epoch+1, args.epochs, iteration, loss.item()))

                if args.conditional:
                    c = c_test[np.random.randint(0, len(c_test)-1, 1), :]
                    c = torch.from_numpy(c).float()
                    x = vae.inference(n=1, c=c)
                else:
                    x = vae.inference(n=1)

                if iteration % 20000 == 0:
                    torch.save(vae.state_dict(), os.path.join(args.model_root, "E{:d}-vae.pt".format(iteration)))

        torch.save(vae.state_dict(), os.path.join(args.model_root, "E{:d}-vae.pt".format(epoch)))


# def test(args):
#     torch.manual_seed(args.seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(args.seed)
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     vae = VAE(
#         latent_size=args.latent_size,
#         conditional=args.conditional,
#         num_labels=10 if args.conditional else 0).to(device)
#
#     vae.load_state_dict(torch.load(args.model_file))
#     vae.eval()
#
#     for i in range(8):
#         if args.conditional:
#             c = torch.arange(0, 10).long().unsqueeze(1)
#             x = vae.inference(n=c.size(0), c=c)
#         else:
#             x = vae.inference(n=10)
#
#         plt.figure()
#         plt.figure(figsize=(5, 10))
#         for p in range(10):
#             plt.subplot(5, 2, p + 1)
#             if args.conditional:
#                 plt.text(
#                     0, 0, "c={:d}".format(c[p].item()), color='black',
#                     backgroundcolor='white', fontsize=8)
#             plt.imshow(x[p].view(28, 28).data.numpy())
#             plt.axis('off')
#         plt.show("Test {:d}".format(i))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--encoder_layer_sizes", type=list, default=[X_dim, h_Q_dim, h_Q_dim])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[h_P_dim, h_P_dim, X_dim])
    parser.add_argument("--latent_size", type=int, default=z_dim)
    parser.add_argument("--print_every", type=int, default=1000)
    parser.add_argument("--fig_root", type=str, default='figs')
    parser.add_argument("--model_root", type=str, default='models')
    parser.add_argument("--conditional", action='store_true', default=True)
    parser.add_argument("--condition_dim", type=int, default=c_dim)

    parser.add_argument("--model_file", type=str, default='models/E4-vae.pt')

    args = parser.parse_args()

    main(args)