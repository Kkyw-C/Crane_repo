# -*- coding: utf-8 -*-
'''
TODO:LQD
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FC
from PyQt5.QtWidgets import * #QApplication, QPushButton, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget
from PyQt5 import  QtCore
from PyQt5.QtGui import QIntValidator
import os
import pickle
import time
import xlrd

import torch
from models import Obs2QmapModel, Obs2QmapModelwithRecon


class QtDraw(QMainWindow):
    #flag_btn_start = True

    def __init__(self):
        super(QtDraw, self).__init__()
        self.load_common_data()
        self.load_local_data(idx=self.cur_lift_obj_pose)
        self.init_ui()
        self.visualize_data()

        self.model = Obs2QmapModel(56*56, 56*56)
        self.model.load_state_dict(torch.load('th_models/init/Obs2QmapModel-225000.pt'))

        self.canvas.draw()  # 这里开始绘制

    def init_ui(self):
        self.resize(1000, 900)
        self.setWindowTitle('Generate Planning Data')

        # TODO:这里是结合的关键
        self.fig = plt.Figure(figsize=(8, 10))
        self.fig.subplots_adjust(left=0.1, right=0.9, top=0.95, bottom=0.05)
        self.canvas = FC(self.fig)
        self.ax = self.fig.add_subplot(111)

        self.ax.set_aspect('equal')  # 为了绘制的图形不变形

        # GUI控件
        # 显示当前任务控件
        self.lb_task = QLabel()
        self.lb_task.setText("")

        # 显示提示信息
        self.lb_tips = QLabel()
        self.lb_tips.setWordWrap(True)
        self.lb_tips.setText("")

        # 保存高斯分布参数的控件
        self.btn_gen_scene_file = QPushButton(self)
        self.btn_gen_scene_file.setText('生成.scene文件')
        self.btn_gen_scene_file.clicked.connect(self.slot_btn_gen_scene_file)
        self.btn_gen_local_map = QPushButton(self)
        self.btn_gen_local_map.setText('生成局部地图文件')
        self.btn_gen_local_map.clicked.connect(self.slot_btn_gen_local_maps)
        self.btn_gen_local_qmap = QPushButton(self)
        self.btn_gen_local_qmap.setText('生成站位质量图文件')
        self.btn_gen_local_qmap.clicked.connect(self.slot_btn_gen_local_qmaps)
        self.btn_pre_task = QPushButton(self)
        self.btn_gen_planning_data = QPushButton(self)
        self.btn_gen_planning_data.setText('生成规划数据')
        self.btn_gen_planning_data.clicked.connect(self.slot_btn_gen_planning_data)
        self.rb_from_file = QRadioButton('检查零散文件', self)
        self.rb_from_file.setChecked(True)
        self.rb_from_dict = QRadioButton('检查生成的数据', self)
        self.bg = QButtonGroup(self)
        self.bg.addButton(self.rb_from_file, 1)
        self.bg.addButton(self.rb_from_dict, 0)
        self.bg.buttonToggled.connect(self.slot_rb_clicked)
        self.cb_show_lift_obj_poses = QCheckBox('被吊物位置', self)
        self.cb_show_lift_obj_poses.setChecked(True)
        self.cb_show_lift_obj_poses.stateChanged.connect(self.slot_btn_show)
        self.cb_show_local_map = QCheckBox('局部地图', self)
        self.cb_show_local_map.setChecked(True)
        self.cb_show_local_map.stateChanged.connect(self.slot_btn_show)
        self.cb_show_local_confs = QCheckBox('无碰撞位形', self)
        self.cb_show_local_confs.setChecked(True)
        self.cb_show_local_confs.stateChanged.connect(self.slot_btn_show)
        self.cb_show_local_qmap = QCheckBox('站位质量图', self)
        self.cb_show_local_qmap.setChecked(True)
        self.cb_show_local_qmap.stateChanged.connect(self.slot_btn_show)
        self.cb_show_pred_local_qmap = QCheckBox('预测站位质量图', self)
        self.cb_show_pred_local_qmap.setChecked(True)
        self.cb_show_pred_local_qmap.stateChanged.connect(self.slot_btn_show)
        self.show_btn_layout = QGridLayout()
        self.show_btn_layout.addWidget(self.cb_show_lift_obj_poses, 0, 0)
        self.show_btn_layout.addWidget(self.cb_show_local_map, 0, 1)
        self.show_btn_layout.addWidget(self.cb_show_local_confs, 1, 0)
        self.show_btn_layout.addWidget(self.cb_show_local_qmap, 1, 1)
        self.show_btn_layout.addWidget(self.cb_show_pred_local_qmap, 2, 0)
        self.show_group_box = QGroupBox('显示选项')
        self.show_group_box.setLayout(self.show_btn_layout)
        self.btn_pre_task.setText('前一任务')
        self.btn_pre_task.clicked.connect(self.slot_btn_pre)
        self.btn_next_task = QPushButton(self)
        self.btn_next_task.setText('下一任务')
        self.btn_next_task.clicked.connect(self.slot_btn_next)
        pre_next_btn_layout = QHBoxLayout()
        pre_next_btn_layout.addWidget(self.btn_pre_task)
        pre_next_btn_layout.addWidget(self.btn_next_task)

        self.btn_test_planning_data = QPushButton(self)
        self.btn_test_planning_data.setText('测试规划数据')
        self.btn_test_planning_data.clicked.connect(self.test_planning_data)

        vLayout = QVBoxLayout()
        vLayout.addWidget(self.lb_task)
        vLayout.addWidget(self.btn_gen_scene_file)
        vLayout.addWidget(self.btn_gen_local_map)
        vLayout.addWidget(self.btn_gen_local_qmap)
        vLayout.addWidget(self.btn_gen_planning_data)
        vLayout.addWidget(self.rb_from_file)
        vLayout.addWidget(self.rb_from_dict)
        vLayout.addWidget(self.show_group_box)
        vLayout.addLayout(pre_next_btn_layout)
        vLayout.addWidget(self.btn_test_planning_data)
        vLayout.addWidget(self.lb_tips)
        vLayout.setAlignment(QtCore.Qt.AlignTop)
        hLayout = QHBoxLayout()
        hLayout.addWidget(self.canvas)
        hLayout.addLayout(vLayout)

        widget = QWidget()
        widget.setLayout(hLayout)
        self.setCentralWidget(widget)

    def load_common_data(self):
        self.local_map_size_ = 56
        self.local_map_n_ = 56
        self.scene_size_ = 250  # 场景正方形的大小(m)
        self.scene_n_ = 250  # 场景栅格数量
        self.obs_boxes = []
        self.obs_cylinders = []
        self.h_max = 0.0

        self.work_space = "./crane_planning_data/"

        # 加载全局地图
        global_map_file = self.work_space + "global_bin_map.txt"
        if os.path.exists(global_map_file):
            self.global_map = np.loadtxt(global_map_file)

        # 加载被吊物位姿
        lift_obj_pos_file = self.work_space + "lift_obj_poses.txt"
        self.lift_obj_poses = np.loadtxt(lift_obj_pos_file, delimiter=',')
        self.cur_lift_obj_pose = 0

        # 局部地图
        self.local_map = []
        self.local_qmap = []
        self.pred_local_qmap = []


        self.local_data_from_file = True
        self.show_lift_obj_poses = True
        self.show_local_map = True
        self.show_local_confs = True
        self.show_local_qmap = True
        self.show_pred_local_qmap = True


        self.planning_data_dict = {}
        # 加载规划数据
        planning_data_dict_file = self.work_space + "planning_data_qmap.pkl"
        if os.path.exists(planning_data_dict_file):
            with open(planning_data_dict_file, "rb") as data_f:
                self.planning_data_dict = pickle.load(data_f)


    def load_local_data(self, idx):
        lift_obj_pose = self.lift_obj_poses[idx]
        print("lift_obj_pose: ", lift_obj_pose)
        lift_obj_pose_str = "({:d},{:d},{:d})".format(int(lift_obj_pose[0]), int(lift_obj_pose[1]),
                                                      int(lift_obj_pose[2]))
        if self.local_data_from_file == 1:
            # 加载局部地图
            local_map_file = self.work_space + "local_obs/" + lift_obj_pose_str + "_local_gray_map.txt"
            if os.path.exists(local_map_file):
                self.local_map = np.loadtxt(local_map_file)

            # 加载起重机器人局部无碰撞站位区
            local_confs_file = self.work_space + "local_confs/" + lift_obj_pose_str + "_confs_free.txt"
            self.local_confs = np.loadtxt(local_confs_file, delimiter=',', usecols=range(2))

            # 加载起重机器人局部站位质量图
            local_qmap_file = self.work_space + "local_qmap/" + lift_obj_pose_str + "_local_qmap.txt"
            if os.path.exists(local_qmap_file):
                self.local_qmap = np.loadtxt(local_qmap_file)
        else:
            print("In DICT")
            # 局部地图
            lift_obj_pose_key = "lift_obj_pose_" + str(idx)
            item = self.planning_data_dict[lift_obj_pose_key]
            self.local_map = item['obs']
            self.local_qmap = item['location_quality_map']
            obs_in = self.local_map.flatten()

            obs_in = torch.from_numpy(obs_in).float()
            pred_qmap = self.model.forward(obs_in)
            pred_qmap = pred_qmap.int().detach().numpy()
            self.pred_local_qmap = np.reshape(pred_qmap, (56, 56))

            # traced_pred_qmap = self.traced_model.forward(obs_in)
            # traced_pred_qmap = traced_pred_qmap.int().detach().numpy()
            # self.traced_pred_qmap = np.reshape(traced_pred_qmap, (56, 56))

            # 加载起重机器人局部无碰撞站位区
            local_confs_file = self.work_space + "local_confs/" + lift_obj_pose_str + "_confs_free.txt"
            self.local_confs = np.loadtxt(local_confs_file, delimiter=',', usecols=range(2))


    def visualize_global_map(self, color):
        rows = int(len(self.global_map))
        x, y = np.nonzero(self.global_map)
        x = x * self.scene_size_ / rows - 0.5 * self.scene_size_
        y = y * self.scene_size_ / rows - 0.5 * self.scene_size_
        self.ax.scatter(x, y, color=color, s=5, alpha=0.9)

    def visualize_local_map(self, lift_obj_pose, local_map, color):
        if local_map == []:
            return

        rows = int(len(local_map))
        x, y = np.nonzero(local_map)
        x = x * self.local_map_size_ / rows - 0.5 * self.local_map_size_ + lift_obj_pose[0]
        y = y * self.local_map_size_ / rows - 0.5 * self.local_map_size_ + lift_obj_pose[1]

        self.ax.scatter(x, y, color=color, s=5, alpha=0.3)

    def visualize_local_confs(self, color):
        self.ax.scatter(self.local_confs[:,0], self.local_confs[:,1], color=color, s=1, alpha=0.3)


    def visualize_data(self):
        self.ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        # 绘制全局地图
        self.visualize_global_map('gray')

        # 绘制被吊物位姿
        if self.show_lift_obj_poses:
            self.ax.scatter(self.lift_obj_poses[:, 0], self.lift_obj_poses[:, 1], color='blue', s=4, alpha=0.8)

        # 绘制当前的被吊物位姿
        self.ax.scatter(self.lift_obj_poses[self.cur_lift_obj_pose, 0], self.lift_obj_poses[self.cur_lift_obj_pose, 1], color='red', s=20, alpha=0.3)

        # 绘制局部地图
        if self.show_local_map:
            self.visualize_local_map(self.lift_obj_poses[self.cur_lift_obj_pose], self.local_map, 'yellow')

        # 绘制起重机器人潜在站位区
        if self.show_local_confs:
            self.visualize_local_confs('green')

        # 绘制站位质量图
        if self.show_local_qmap:
            self.visualize_local_map(self.lift_obj_poses[self.cur_lift_obj_pose], self.local_qmap, 'red')

        if self.show_pred_local_qmap:
            self.visualize_local_map(self.lift_obj_poses[self.cur_lift_obj_pose], self.pred_local_qmap, 'magenta')

        self.canvas.draw()  # 这里开始绘制

    def generate_scene_file(self):
        obs_file = self.work_space + "obs.xlsx"
        data = xlrd.open_workbook(obs_file)  # 打开xlsx文件
        scene_file = self.work_space + "obs_{:d}x{:d}.scene".format(self.scene_n_, self.scene_n_)
        file = open(scene_file, 'w')
        file.write('scene\n')
        names = data.sheet_names()

        # 长方体障碍物
        if('boxes' in names):
            # table = data.sheets()[0]  # 打开第一张表
            table = data.sheet_by_name("boxes")  # 打开名为“boxes”的表单
            nrows = table.nrows  # 获取表的行数
            for i in range(nrows):
                if (i == 0):  # 跳过第一行的标题
                    continue
                file.write('* box' + str(i) + '\n')  # 几何体名字
                file.write('1\n')
                file.write('box\n')  # 几何体类型
                for k in range(0, 3):  # 长方体的长宽高
                    file.write('%.2f' % table.cell(i, k).value + ' ')
                file.write('\n')
                for k in range(3, 5):  # 长方体的形心坐标
                    file.write('%.2f' % table.cell(i, k).value + ' ')
                file.write('%.2f' % (table.cell(i, 5).value + 0.5 * table.cell(i, 2).value) + '\n')
                file.write('0 0 0 1\n')  # 长方体的朝向，现在写死
                file.write('0 0 0 0\n')


        # 圆柱体障碍物
        if('cylinders' in names):
            # table = data.sheets()[0]  # 打开第一张表
            table = data.sheet_by_name("cylinders")  # 打开名为“cylinders”的表单
            nrows = table.nrows  # 获取表的行数
            for i in range(nrows):
                if (i == 0):  # 跳过第一行的标题
                    continue
                file.write('* cylinder' + str(i) + '\n')  # 几何体名字
                file.write('1\n')
                file.write('cylinder\n')  # 几何体类型
                for k in range(0, 2):  # 圆柱体半径和高
                    file.write('%.2f' % table.cell(i, k).value + ' ')
                file.write('\n')
                for k in range(2, 4):  # 圆柱体的形心坐标
                    file.write('%.2f' % table.cell(i, k).value + ' ')
                file.write('%.2f' % (table.cell(i, 4).value + 0.5 * table.cell(i, 1).value) + '\n')
                # print('%.2f' % (table.cell(i, 5).value + 0.5*table.cell(i, 2).value) + '\n')
                file.write('0 0 0 1\n')  # 长方体的朝向，现在写死
                file.write('0 0 0 0\n')

        file.write('.\n')
        file.close()

        print("场景文件已保存为：%s", scene_file)


    def generate_local_bin_and_gray_map(self, map_size, num, out_dir):
        # 读取excel
        self.read_excel()

        step_x = step_y = map_size / num

        # 创建dict
        bin_map_dict = {}
        gray_map_dict = {}

        for obj_idx in range(len(self.lift_obj_poses)):
            print("第", obj_idx, "个被吊物位置：", self.lift_obj_poses[obj_idx])
            # 创建二维数组
            bin_map = np.zeros((num, num), dtype=np.int)
            gray_map = np.zeros((num, num), dtype=np.int)
            # 每个小格分别与障碍物进行判断，看是否重合相交
            for i in range(num):
                for j in range(num):
                    grid_x_min = i * step_x - 0.5 * map_size + self.lift_obj_poses[obj_idx][0]
                    grid_x_max = (i + 1) * step_x - 0.5 * map_size + self.lift_obj_poses[obj_idx][0]
                    grid_y_min = j * step_y - 0.5 * map_size + self.lift_obj_poses[obj_idx][1]
                    grid_y_max = (j + 1) * step_y - 0.5 * map_size + self.lift_obj_poses[obj_idx][1]
                    # 与所有长方体进行相交检测
                    box_num = len(self.obs_boxes)  # 长方体个数
                    for k in range(box_num):  # self.obs_boxes[k] = [长,宽,高,x,y,z,theta]
                        obs_x_min = self.obs_boxes[k][3] - 0.5 * self.obs_boxes[k][0]
                        obs_x_max = self.obs_boxes[k][3] + 0.5 * self.obs_boxes[k][0]
                        obs_y_min = self.obs_boxes[k][4] - 0.5 * self.obs_boxes[k][1]
                        obs_y_max = self.obs_boxes[k][4] + 0.5 * self.obs_boxes[k][1]
                        if (
                                grid_x_min < obs_x_max and grid_x_max > obs_x_min and grid_y_min < obs_y_max and grid_y_max > obs_y_min):
                            bin_map[i][j] = 1
                            gray_map[i][j] = 255 * self.obs_boxes[k][2] / self.h_max
                            #plt.scatter(i,j,s=10,c='blue')

                    # 与所有圆柱体进行相交检测
                    grid_org = np.array([0.5 * (grid_x_min + grid_x_max), 0.5 * (grid_y_min + grid_y_max)])  # 栅格中心坐标
                    grid_right_top = grid_org + np.array([0.5 * step_x, 0.5 * step_y])
                    a = grid_right_top - grid_org
                    cylinder_num = len(self.obs_cylinders)  # 长方体个数
                    for k in range(cylinder_num):  # self.obs_boxes[k] = [半径,高,x,y,z]
                        b = np.array([np.fabs(self.obs_cylinders[k][2] - grid_org[0]),
                                      np.fabs(self.obs_cylinders[k][3] - grid_org[1])])
                        c = b - a
                        if (c[0] < 0):
                            c[0] = 0
                        if (c[1] < 0):
                            c[1] = 0

                        if (np.linalg.norm(c) < self.obs_cylinders[k][0]):
                            bin_map[i][j] = 1
                            gray_map[i][j] = 255 * self.obs_boxes[k][2] / self.h_max


            lift_obj_pose_str = "({:d},{:d},{:d})".format(int(self.lift_obj_poses[obj_idx][0]),
                                                          int(self.lift_obj_poses[obj_idx][1]),
                                                          int(self.lift_obj_poses[obj_idx][2]))
            bin_map_file = lift_obj_pose_str + "_local_bin_map.txt"
            gray_map_file = lift_obj_pose_str + "_local_gray_map.txt"
            np.savetxt(out_dir + bin_map_file, bin_map, fmt="%d")
            np.savetxt(out_dir + gray_map_file, gray_map, fmt="%d")
            bin_map_dict[lift_obj_pose_str] = bin_map
            gray_map_dict[lift_obj_pose_str] = gray_map

        # # 存储数据
        # bin_map_file_name = out_dir + "/local_bin_map_dict" + "_{:d}x{:d}.pkl".format(num, num)
        # gray_map_file_name = out_dir + "/local_gray_map_dict" + "_{:d}x{:d}.pkl".format(num, num)
        # with open(bin_map_file_name, "wb") as bin_f:
        #     pickle.dump(bin_map_dict, bin_f)
        # with open(gray_map_file_name, "wb") as gray_f:
        #     pickle.dump(gray_map_dict, gray_f)

    def read_excel(self):
        obs_file = self.work_space + "obs.xlsx"
        workbook = xlrd.open_workbook(obs_file)  # 获取所有sheet
        names = workbook.sheet_names()

        if ('boxes' in names):
            table = workbook.sheet_by_name("boxes")  # 打开名为“boxes”的表单
            nrows = table.nrows  # 获取表的行数
            for i in range(1, nrows):
                self.obs_boxes.append(table.row_values(i))
                if(self.h_max < table.row_values(i)[2]):
                    self.h_max = table.row_values(i)[2]

        if ('cylinders' in names):
            table = workbook.sheet_by_name("cylinders")  # 打开名为“boxes”的表单
            nrows = table.nrows  # 获取表的行数
            for i in range(1, nrows):
                self.obs_cylinders.append(table.row_values(i))
                if(self.h_max < table.row_values(i)[1]):
                    self.h_max = table.row_values(i)[1]


    def generate_qmap_files(self):
        num = len(self.lift_obj_poses)
        for i in range(num):
            print("第", i, "个被吊物位置：", self.lift_obj_poses[i])
            lift_obj_pose = self.lift_obj_poses[i]

            lift_obj_pose_str = "({:d},{:d},{:d})".format(int(lift_obj_pose[0]), int(lift_obj_pose[1]),
                                                          int(lift_obj_pose[2]))

            confs = np.loadtxt(self.work_space + "local_confs/" + lift_obj_pose_str + "_confs_free.txt", delimiter=',',
                               usecols=range(2))
            self.ax.scatter(confs[:, 0], confs[:, 1], color='green', s=1, alpha=0.3)

            # 计算相对坐标，以被吊物位置为坐标原点
            confs[:, 0] -= lift_obj_pose[0]
            confs[:, 1] -= lift_obj_pose[1]
            # 将坐标原点平移到质量图的左下角，使得坐标值均为正，便于计算质量图
            confs[:, 0] += int(0.5*self.local_map_n_)
            confs[:, 1] += int(0.5*self.local_map_n_)
            location_quality_map = np.zeros((self.local_map_n_, self.local_map_n_), dtype=np.int)
            n = len(confs)
            for k in range(n):
                x_idx = int(confs[k][0])
                y_idx = int(confs[k][1])
                location_quality_map[x_idx][y_idx] += 1

            lift_obj_pose_str = "({:d},{:d},{:d})".format(int(self.lift_obj_poses[i][0]),
                                                          int(self.lift_obj_poses[i][1]),
                                                          int(self.lift_obj_poses[i][2]))
            qmap_file = self.work_space + "local_qmap/" + lift_obj_pose_str + "_local_qmap.txt"
            np.savetxt(qmap_file, location_quality_map, fmt="%d")


    def generate_planning_data(self):
        train_data = {}
        num = len(self.lift_obj_poses)
        for i in range(num):
            print("第", i, "个被吊物位置：", self.lift_obj_poses[i])
            lift_obj_pose = self.lift_obj_poses[i]

            lift_obj_pose_key = "lift_obj_pose_{:d}".format(i)

            lift_obj_pose_str = "({:d},{:d},{:d})".format(int(lift_obj_pose[0]), int(lift_obj_pose[1]),
                                                          int(lift_obj_pose[2]))

            confs = np.loadtxt(self.work_space + "local_confs/" + lift_obj_pose_str + "_confs_free.txt", delimiter=',',
                               usecols=range(2))
            self.ax.scatter(confs[:, 0], confs[:, 1], color='green', s=1, alpha=0.3)

            # 计算相对坐标，以被吊物位置为坐标原点
            confs[:, 0] -= lift_obj_pose[0]
            confs[:, 1] -= lift_obj_pose[1]
            # 将坐标原点平移到质量图的左下角，使得坐标值均为正，便于计算质量图
            confs[:, 0] += int(0.5 * self.local_map_n_)
            confs[:, 1] += int(0.5 * self.local_map_n_)
            location_quality_map = np.zeros((self.local_map_n_, self.local_map_n_), dtype=np.int)
            n = len(confs)
            for k in range(n):
                x_idx = int(confs[k][0])
                y_idx = int(confs[k][1])
                location_quality_map[x_idx][y_idx] += 1

            train_data[lift_obj_pose_key] = {}
            train_data[lift_obj_pose_key]['lift_obj_pose_name'] = lift_obj_pose_str
            train_data[lift_obj_pose_key]['lift_obj_pose'] = lift_obj_pose
            local_map = np.loadtxt(self.work_space + "local_obs/" + lift_obj_pose_str + "_local_gray_map.txt")
            train_data[lift_obj_pose_key]['obs'] = local_map
            train_data[lift_obj_pose_key]['location_quality_map'] = location_quality_map

        planning_data_file = self.work_space + "planning_data_qmap.pkl"
        with open(planning_data_file, "wb") as p_f:
            pickle.dump(train_data, p_f)


    def test_planning_data(self):
        planning_data_dict_file = self.work_space + "planning_data_qmap.pkl"
        if os.path.exists(planning_data_dict_file):
            with open(planning_data_dict_file, "rb") as data_f:
                self.planning_data_dict = pickle.load(data_f)

            self.cur_lift_obj_pose = 0
            self.rb_from_dict.setChecked(True)
            self.load_local_data(self.cur_lift_obj_pose)

    def slot_btn_gen_planning_data(self):
        self.generate_planning_data()
        self.lb_tips.setText("规划数据已生成，保存到：{:s}".format(self.work_space))

    def slot_btn_pre(self):
        if self.cur_lift_obj_pose - 1 >= 0:
            self.cur_lift_obj_pose -= 1
            print("切换到：", self.cur_lift_obj_pose)
            self.load_local_data(self.cur_lift_obj_pose)
            self.visualize_data()

    def slot_btn_next(self):
        if self.cur_lift_obj_pose + 1 < len(self.lift_obj_poses):
            self.cur_lift_obj_pose += 1
            print("切换到：", self.cur_lift_obj_pose)
            self.load_local_data(self.cur_lift_obj_pose)
            self.visualize_data()


    def slot_btn_gen_scene_file(self):
        self.generate_scene_file()
        self.lb_tips.setText("场景文件已生成，保存到：{:s}".format(self.work_space))

    def slot_btn_gen_local_maps(self):
        out_file = self.work_space + "local_obs/"
        self.generate_local_bin_and_gray_map(56, 56, out_file)
        tips = "局部地图文件已生成，保存到：{:s}".format(self.work_space) + "local_obs"
        self.lb_tips.setText(tips)

    def slot_btn_gen_local_qmaps(self):
        self.generate_qmap_files()

    def slot_rb_clicked(self):
        self.local_data_from_file = self.bg.checkedId()

    def slot_btn_show(self):
        self.show_lift_obj_poses = self.cb_show_lift_obj_poses.checkState() == QtCore.Qt.Checked
        self.show_local_map = self.cb_show_local_map.checkState() == QtCore.Qt.Checked
        self.show_local_confs = self.cb_show_local_confs.checkState() == QtCore.Qt.Checked
        self.show_local_qmap = self.cb_show_local_qmap.checkState() == QtCore.Qt.Checked
        self.show_pred_local_qmap = self.cb_show_pred_local_qmap.checkState() == QtCore.Qt.Checked
        self.visualize_data()


def ui_main():
    app = QApplication(sys.argv)
    w = QtDraw()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    ui_main()