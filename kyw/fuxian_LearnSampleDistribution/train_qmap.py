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

def get_obs_qmap_dataset(data_file, shuffle=False):
    with open(data_file, "rb") as data_f:
        raw_data = pickle.load(data_f)

    dataset = {}
    data = []
    num = len(raw_data)
    for i in range(num):
        item_id = "lift_obj_pose_" + str(i)
        item_data = raw_data[item_id]
        lift_obj_pose = item_data['lift_obj_pose']
        obs = item_data['obs']
        qmap = item_data['location_quality_map']
        data.append(np.concatenate([lift_obj_pose, obs.flatten(), qmap.flatten()], np.newaxis))

    data = np.array(data)
    print(data.shape)
    if shuffle:
        np.random.shuffle(data)

    # 组装训练集、校验集、测试集
    local_map_size = 56
    test_size = val_size = int(0.05*num)
    test_data = data[0:test_size, :]
    val_data = data[test_size:test_size+val_size, :]
    train_data = data[test_size+val_size:num, :]
    #print("train_data size:", len(train_data))
    dataset['test'] = dataset['val'] = dataset['train'] = {}
    dataset['test']['lift_obj_pose'] = test_data[:, 0:3]
    dataset['test']['obs'] = test_data[:, 3:3+local_map_size*local_map_size]
    dataset['test']['qmap'] = test_data[:, 3+local_map_size*local_map_size:3+2*local_map_size*local_map_size]
    dataset['val']['lift_obj_pose'] = val_data[:, 0:3]
    dataset['val']['obs'] = val_data[:, 3:3 + local_map_size * local_map_size]
    dataset['val']['qmap'] = val_data[:, 3+local_map_size*local_map_size:3+2*local_map_size*local_map_size]
    dataset['train']['lift_obj_pose'] = train_data[:, 0:3]
    dataset['train']['obs'] = train_data[:, 3:3+local_map_size*local_map_size]
    dataset['train']['qmap'] = train_data[:, 3+local_map_size*local_map_size:3+2*local_map_size*local_map_size]

    #print("train_data size:", len(dataset['train']['obs']))

    # 保存数据集
    ds_file = 'runs/qmap_logs/dataset.pkl'
    with open(ds_file, "wb") as ds_f:
        pickle.dump(dataset, ds_f)

    return dataset

vt = PlanningDataVisualTool()
def visualize_qmap(lift_obj_pose, obs, qmap_target, qmap, it, prefix, i = 0):
    plt.axes().set_aspect('equal')  # 为了绘制的图形不变形
    plt.title('Lifting Scene')
    plt.xlabel('X')
    plt.ylabel('Y')

    vt.visualize_global_map('gray')

    plt.scatter(lift_obj_pose[0], lift_obj_pose[1], s= 20, c='red')

    obs = np.reshape(obs, (56, 56))
    vt.visualize_local_map(lift_obj_pose, obs, 'blue', flag=True)

    qmap_target = np.reshape(qmap_target, (56, 56))
    vt.visualize_local_map(lift_obj_pose, qmap_target, 'green', flag=True)

    qmap = np.reshape(qmap, (56, 56))
    vt.visualize_local_map(lift_obj_pose, qmap, 'red', flag=True)

    plt.savefig("figs/{:s}/{:d}_qmap-fig-{:d}.png".format(prefix, i+1, it), dpi=300)

    #plt.show()

    plt.clf()
    plt.close()

def test_dataset():
    ds = get_obs_qmap_dataset("./crane_planning_data/planning_data_qmap.pkl", shuffle=True)
    num = len(ds['test']['obs'])
    print(num)
    pred_qmap = np.zeros((56, 56))
    for i in range(num):
        lift_obj_pose = ds['test']['lift_obj_pose'][i]
        obs = ds['test']['obs'][i]
        qmap = ds['test']['qmap'][i]
        visualize_qmap(lift_obj_pose, obs, qmap, pred_qmap, i, "kk")


#test_dataset()

def test(model, test_dataset, it, prefix, random_one = True):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_num = len(test_dataset['obs'])
    if random_one:
        #idx = randint(0, test_num-1)
        idx = [1, 5, 8]
        test_obs = test_dataset['obs'][idx]
        test_qmap = test_dataset['qmap'][idx]
        test_lift_obj_pose = test_dataset['lift_obj_pose'][idx]
        test_obs = torch.from_numpy(test_obs).float().to(device)
        pred_qmap = model.forward(test_obs)

        test_obs = test_obs.detach().numpy()
        pred_qmap = pred_qmap.detach().numpy()

        # print("lift_obj_pose: ", test_lift_obj_pose)
        # print("obs: ", test_obs)
        # print("test_qmap: ", test_qmap)
        # print("pred_qmap: ", pred_qmap)

        # lift_obj_pose_str = "_({:d},{:d},{:d})".format(int(test_lift_obj_pose[0]), int(test_lift_obj_pose[1]), int(test_lift_obj_pose[2]))
        # np.savetxt("./tmp/" + str(it) + lift_obj_pose_str + "_target_qmap.txt", np.reshape(np.array(test_qmap), (56, 56)), fmt="%d")
        # np.savetxt("./tmp/" + str(it) + lift_obj_pose_str + "_pred_qmap.txt", np.reshape(np.array(pred_qmap), (56, 56)), fmt='%.1f')
        # np.savetxt("./tmp/" + str(it) + lift_obj_pose_str + "_diff_qmap.txt", np.reshape(np.array(abs(pred_qmap-test_qmap)), (56, 56)), fmt='%.1f')
        #
        # visualize_qmap(test_lift_obj_pose, test_obs, test_qmap, pred_qmap, it, prefix)
        for i in range(3):
            lift_obj_pose_str = "_({:d},{:d},{:d})".format(int(test_lift_obj_pose[i][0]), int(test_lift_obj_pose[i][1]),
                                                           int(test_lift_obj_pose[i][2]))
            np.savetxt("./tmp/" + str(i+1) + "_" + str(it) + lift_obj_pose_str + "_target_qmap.txt",
                       np.reshape(np.array(test_qmap[i]), (56, 56)), fmt="%d")
            np.savetxt("./tmp/" + str(i+1) + "_" + str(it) + lift_obj_pose_str + "_pred_qmap.txt",
                       np.reshape(np.array(pred_qmap[i]), (56, 56)), fmt='%.1f')
            np.savetxt("./tmp/" + str(i+1) + "_" + str(it) + lift_obj_pose_str + "_diff_qmap.txt",
                       np.reshape(np.array(abs(pred_qmap[i] - test_qmap[i])), (56, 56)), fmt='%.1f')

            visualize_qmap(test_lift_obj_pose[i], test_obs[i], test_qmap[i], pred_qmap[i], it, prefix, i)

    else:
        test_obs = test_dataset['obs']
        test_qmap = test_dataset['qmap']
        pred_qmap = model.forward(test_obs)


def train(model, dataset):
    obs_dim = 56 * 56
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=10e-6)

    global_it = 0
    mb_size = 128
    EPOCHS = 20000
    num_train = len(dataset['train']['obs'])
    BATCHES = num_train // mb_size + 1
    running_loss = 0.0
    for epoch in range(EPOCHS):
        model.train()
        for i in range(BATCHES):
            if i * mb_size < num_train:
                mb_begin = i * mb_size
                if mb_begin + mb_size < num_train:
                    mb_end = mb_begin + mb_size
                else:
                    mb_end = num_train
            else:
                break
            mb_obs = dataset['train']['obs'][mb_begin:mb_end, :]
            mb_qmap = dataset['train']['qmap'][mb_begin:mb_end, :]
            mb_obs = torch.from_numpy(mb_obs).float().to(device)
            mb_qmap = torch.from_numpy(mb_qmap).float().to(device)

            pred_qmap = model(mb_obs)
            loss = model.loss(pred_qmap, mb_qmap)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if global_it == 0:
                writer.add_scalar('training loss', loss.item(), global_it)
            running_loss += loss.item()
            if global_it % 500 == 499:
                print('Iter: {}'.format(global_it))
                print('Loss: {:.4}'.format(loss))
                print()
                writer.add_scalar('training loss', running_loss / 500.0, global_it)
                running_loss = 0.0

                test(model, dataset['val'], global_it, "init")

                # 用验证集所有数据来验证
                val_obs = dataset['val']['obs']
                val_qmap = dataset['val']['qmap']
                val_obs = torch.from_numpy(val_obs).float().to(device)
                val_qmap = torch.from_numpy(val_qmap).float().to(device)
                pred_val_qmap = model(val_obs)
                val_loss = model.loss(pred_val_qmap, val_qmap)
                writer.add_scalar('validating loss', val_loss.item(), global_it)

                # 模型的准确度或性能如何度量？？？？
                # 1）红色和绿色覆盖度；2）有些点好，有些点不好，怎么统计。

            if global_it % 5000 == 0 and global_it != 0:
                torch.save(model.state_dict(),
                           os.path.join("th_models", "init", "Obs2QmapModel-{:d}.pt".format(global_it)))

            global_it += 1



ds = get_obs_qmap_dataset("./crane_planning_data/planning_data_qmap.pkl", shuffle=True)
# obs_dim = qmap_dim = 56 * 56
# model = Obs2QmapModel(obs_dim, qmap_dim)
# train(model, ds)
#
# # 测试
# test_obs = ds['test']['obs']
# test_qmap = ds['test']['qmap']
# test_obs = torch.from_numpy(test_obs).float()
# test_qmap = torch.from_numpy(test_qmap).float()
# pred_val_qmap = model(test_obs)
# test_loss = model.loss(pred_val_qmap, test_qmap)
# writer.add_scalar('test loss', test_loss.item())
#
# model.load_state_dict(torch.load('th_models/init/Obs2QmapModel-225000.pt'))
# for i in range(200):
#     test(model, ds['train'], i, "goal")


# obs_dim = qmap_dim = 56 * 56
# model = Obs2QmapModel(obs_dim, qmap_dim)
# model.load_state_dict(torch.load('th_models/init/Obs2QmapModel-225000.pt'))
# input = torch.tensor(ds['test']['obs'][0]).float()
# input = input.view((1, -1))
# traced_script_module = torch.jit.trace(model, input)
# traced_script_module.save("./th_models/init/traced_model.pt")
#
# input = torch.tensor(ds['test']['obs'][1]).float()
# input = input.view((1, -1))
# output = traced_script_module(input)
#
# lift_obj_pose = ds['test']['lift_obj_pose'][0]
# print(lift_obj_pose)
# print(input.numpy().tolist()[0])
# print(output.detach().numpy().tolist()[0])
#
# np.savetxt("lift_obj_pose.txt", lift_obj_pose, fmt='%.2f')
# np.savetxt("obs.txt", input.numpy().tolist()[0], fmt='%d')
# np.savetxt("qmap.txt", ds['test']['qmap'][0], fmt='%d')
# np.savetxt("pred_qmap.txt", output.detach().numpy().tolist()[0], fmt='%.1f')
