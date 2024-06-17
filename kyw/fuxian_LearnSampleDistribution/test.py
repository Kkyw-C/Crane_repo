# -*- coding: UTF-8 -*-

from random import randint, random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import torch

x = [[2.0, 5],
[3, 4],
[5, 4]]
y = [[2, 6.0],
[5, 4],
[5, 4]]
x = torch.from_numpy(np.array(x))
y = torch.from_numpy(np.array(y))

# x = x.flatten()
# y = y.flatten()
w = np.array([2.0, 1])
w = torch.from_numpy(w)
print("w:", w)

s = (x-y)**2
sw = s*w
sum = torch.sum(s)
l = sum / len(x)

loss = torch.nn.functional.mse_loss(x.view(-1, 2), y.view(-1, 2), reduction='mean')
def mse_loss_with_weight(x, y, w):
    se = (x-y)**2
    print("se:", se)
    sew = se * w
    print("sew:", sew)
    mse = torch.mean(sew)
    return mse

l = mse_loss_with_weight(x.view(-1, 2), y.view(-1, 2), w)

print("l:", l)
print("loss:", loss)




# c_test = np.load("c_test.npy")
# c_gapsInitGoal = np.load("c_gapsInitGoal.npy")
# occGridSamples = np.load("occGridSamples.npy")
# numTest = c_test.shape[0]
#
# dimW = 3
# z_dim = 3
# gridSize = 11
# dim = 6
#
# # plot the latent space
# num_viz = 3000
#
# vizIdx = randint(0 ,numTest -1);
# print(vizIdx)
# c_sample_seed = c_test[vizIdx ,:]
# c_sample = np.repeat([c_sample_seed] ,num_viz ,axis=0)
# c_viz = c_gapsInitGoal[vizIdx ,:]
#
#
# from models import VAE
# import torch
#
# vae = VAE(
#         encoder_layer_sizes=[6, 512, 512],
#         latent_size=3,
#         decoder_layer_sizes=[512, 512, 6],
#         conditional=True,
#         condition_dim=133)
#
# vae.load_state_dict(torch.load("models/E480000-vae.pt"))
# vae.eval()
#
# c_sample = torch.from_numpy(c_sample).float()
# y_viz = vae.inference(n=num_viz, c=c_sample)
#
# y_viz = y_viz.detach().numpy()
#
# # # directly sample from the latent space (preferred, what we will use in the end)
# # y_viz, z_viz = sess.run([y, z], feed_dict={z: np.random.randn(num_viz, z_dim), c: c_sample})
#
# fig1 = plt.figure(figsize=(10 ,6), dpi=80)
# ax1 = fig1.add_subplot(111, aspect='equal')
#
# plt.scatter(y_viz[: ,0] ,y_viz[: ,1], color="green", s=70, alpha=0.1)
#
# dw = 0.1
# dimW = 3
# gap1 = c_viz[0:3]
# gap2 = c_viz[3:6]
# gap3 = c_viz[6:9]
# init = c_viz[9:15]
# goal = c_viz[15:21]
#
# obs1 = [0, gap1[1 ] -dw, -0.5,             gap1[0], gap1[1], 1.5]
# obs2 = [gap2[0 ] -dw, 0, -0.5,             gap2[0], gap2[1], 1.5];
# obs3 = [gap2[0 ] -dw, gap2[1 ] +dw, -0.5,    gap2[0], 1, 1.5];
# obs4 = [gap1[0 ] +dw, gap1[1 ] -dw, -0.5,    gap3[0], gap1[1], 1.5];
# obs5 = [gap3[0 ] +dw, gap1[1 ] -dw, -0.5,    1, gap1[1], 1.5];
# obsBounds = [-0.1, -0.1, -0.5, 0, 1.1, 1.5,
#              -0.1, -0.1, -0.5, 1.1, 0, 1.5,
#              -0.1, 1, -0.5, 1.1, 1.1, 1.5,
#              1, -0.1, -0.5, 1.1, 1.1, 1.5 ,]
#
# obs = np.concatenate((obs1, obs2, obs3, obs4, obs5, obsBounds), axis=0)
# for i in range(0 ,int(obs.shape[0] /( 2 *dimW))):
#     ax1.add_patch(
#         patches.Rectangle(
#             (obs[ i * 2 *dimW], obs[ i * 2 *dimW +1]),   # (x,y)
#             obs[ i * 2 *dimW +dimW] - obs[ i * 2 *dimW],          # width
#             obs[ i * 2 *dimW +dimW +1] - obs[ i * 2 *dimW +1],          # height
#             alpha=0.6
#         ))
#
# for i in range(0 ,gridSize *gridSize): # plot occupancy grid
#     cIdx = i + 2* dim
#     if c_sample_seed[cIdx] == 0:
#         plt.scatter(occGridSamples[i, 0], occGridSamples[i, 1], color="red", s=50, alpha=0.7)
#     else:
#         plt.scatter(occGridSamples[i, 0], occGridSamples[i, 1], color="green", s=50, alpha=0.7)
#
# plt.scatter(init[0], init[1], color="red", s=250, edgecolors='black')  # init
# plt.scatter(goal[0], goal[1], color="blue", s=250, edgecolors='black')  # goal
#
# plt.show()
#
# plt.figure(figsize=(10, 6), dpi=80)
# viz1 = 1;
# viz2 = 4;
# plt.scatter(y_viz[:, viz1], y_viz[:, viz2], color="green", s=70, alpha=0.1)
# plt.scatter(c_viz[viz1 + 9], c_viz[viz2 + 9], color="red", s=250, edgecolors='black')  # init
# # saver = tf.train.Saver()
# # saver.save(sess, ‘my_test_model’)
# plt.scatter(c_viz[viz1 + 9 + dim], c_viz[viz2 + 9 + dim], color="blue", s=500, edgecolors='black')  # goal
# plt.show()


#
# import tensorflow as tf
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import matplotlib.gridspec as gridspec
# from mpl_toolkits.mplot3d import Axes3D
# import os
# import csv
# from random import randint, random
# import time
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#
# # (restrict tensorflow memory growth)
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
#
# # neural network parameters
# mb_size = 256
# h_Q_dim = 512
# h_P_dim = 512
#
# c = 0
# lr = 1e-4
#
# # problem dimensions
# dim = 6
# dataElements = dim+3*3+2*dim # sample (6D), gap1 (2D, 1D orientation), gap2, gap3, init (6D), goal (6D)
#
# z_dim = 3 # latent
# X_dim = dim # samples
# y_dim = dim # reconstruction of the original point
# c_dim = dataElements - dim # dimension of conditioning variable
#
# # read in data from csv
# filename = 'narrowDataFile(orginal).txt'
# f = open(filename, 'r')
# reader = csv.reader(f, delimiter=',')
#
# count = 0
# data_list = []
# for row in reader:
#     data_list.append(row[0:dataElements])
# print(data_list)
# data = np.array(data_list,dtype='d')
# numEntries = data.shape[0]
#
# # split the inputs and conditions into test train (to be processed in the next step into an occupancy grid representation)
# ratioTestTrain = 0.8;
# numTrain = int(numEntries*ratioTestTrain)
#
# X_train = data[0:numTrain,0:dim] # state: x, y, z, xdot, ydot, zdot
# c_train = data[0:numTrain,dim:dataElements] # conditions: gaps, init (6), goal (6)
#
# X_test = data[numTrain:numEntries,0:dim]
# c_test = data[numTrain:numEntries,dim:dataElements]
# numTest = X_test.shape[0]
#
# # change conditions to occupancy grid
# def isSampleFree(sample, obs):
#     for o in range(0,int(obs.shape[0]/(2*dimW))):
#         isFree = 0
#         for d in range(0,sample.shape[0]):
#             if (sample[d] < obs[2*dimW*o + d] or sample[d] > obs[2*dimW*o + d + dimW]):
#                 isFree = 1
#                 break
#         if isFree == 0:
#             return 0
#     return 1
#
# gridSize = 11
# dimW = 3
# plotOn = False;
#
# # process data into occupancy grid
# conditions = data[0:numEntries,dim:dataElements]
# conditionsOcc = np.zeros([numEntries,gridSize*gridSize])
# occGridSamples = np.zeros([gridSize*gridSize, 2])
# gridPointsRange = np.linspace(0,1,num=gridSize)
#
# idx = 0;
# for i in gridPointsRange:
#     for j in gridPointsRange:
#         occGridSamples[idx,0] = i
#         occGridSamples[idx,1] = j
#         idx += 1;
#
# start = time.time();
# for j in range(0,numEntries,1):
#     dw = 0.1
#     dimW = 3
#     gap1 = conditions[j,0:3]
#     gap2 = conditions[j,3:6]
#     gap3 = conditions[j,6:9]
#     init = conditions[j,9:15]
#     goal = conditions[j,15:21]
#
#     obs1 = [0, gap1[1]-dw, -0.5,             gap1[0], gap1[1], 1.5]
#     obs2 = [gap2[0]-dw, 0, -0.5,             gap2[0], gap2[1], 1.5];
#     obs3 = [gap2[0]-dw, gap2[1]+dw, -0.5,    gap2[0], 1, 1.5];
#     obs4 = [gap1[0]+dw, gap1[1]-dw, -0.5,    gap3[0], gap1[1], 1.5];
#     obs5 = [gap3[0]+dw, gap1[1]-dw, -0.5,    1, gap1[1], 1.5];
#     obs = np.concatenate((obs1, obs2, obs3, obs4, obs5), axis=0)
#
#     if j % 5000 == 0:
#         print('Iter: {}'.format(j))
#
#     occGrid = np.zeros(gridSize*gridSize)
#     for i in range(0,gridSize*gridSize):
#         occGrid[i] = isSampleFree(occGridSamples[i,:],obs)
#     conditionsOcc[j,:] = occGrid
#
#     if plotOn:
#         fig1 = plt.figure(figsize=(10,6), dpi=80)
#         ax1 = fig1.add_subplot(111, aspect='equal')
#         for i in range(0,obs.shape[0]/(2*dimW)): # plot obstacle patches
#             ax1.add_patch(
#             patches.Rectangle(
#                 (obs[i*2*dimW], obs[i*2*dimW+1]),   # (x,y)
#                 obs[i*2*dimW+dimW] - obs[i*2*dimW],          # width
#                 obs[i*2*dimW+dimW+1] - obs[i*2*dimW+1],          # height
#                 alpha=0.6
#             ))
#         for i in range(0,gridSize*gridSize): # plot occupancy grid
#             if occGrid[i] == 0:
#                 plt.scatter(occGridSamples[i,0], occGridSamples[i,1], color="red", s=70, alpha=0.8)
#             else:
#                 plt.scatter(occGridSamples[i,0], occGridSamples[i,1], color="green", s=70, alpha=0.8)
#         plt.show()
# end = time.time();
# print('Time: ', end-start)
#
# cs = np.concatenate((data[0:numEntries,dim+3*dimW:dataElements], conditionsOcc), axis=1) # occ, init, goal
# c_dim = cs.shape[1]
# c_gapsInitGoal = c_test
# c_train = cs[0:numTrain,:]
# c_test = cs[numTrain:numEntries,:]
#
#
# np.save("X_train", X_train)
# np.save("c_train", c_train)
# np.save("c_test", c_test)
# np.save("c_gapsInitGoal", c_gapsInitGoal)
# np.save("occGridSamples", occGridSamples)

