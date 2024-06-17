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


def clip(lift_obj_pose, data):
    confs = []
    R_min = 10
    R_max = 28
    for i in range(len(data)):
        d = np.sqrt((data[i][0] - lift_obj_pose[0]) ** 2 + (data[i][1] - lift_obj_pose[1]) ** 2)
        if R_min < d and d < R_max:
            confs.append(data[i])
    confs = np.array(confs)
    return confs

class QtDraw(QMainWindow):
    #flag_btn_start = True

    def __init__(self):
        super(QtDraw, self).__init__()
        self.init_ui()
        self.load_common_data()
        self.load_task_data(self.cur_task)

        self.visualize_data()
        self.canvas.draw()  # 这里开始绘制

    def init_ui(self):
        self.resize(1000, 900)
        self.setWindowTitle('Generate Planning Data Using Gaussian Distribution')

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
        # 控制高斯分布均值的spinbox
        self.sp_mean_x = QSpinBox()
        self.sp_mean_x.valueChanged.connect(self.resampling)
        self.sp_mean_x.setRange(-130, 130)
        self.sp_mean_y = QSpinBox()
        self.sp_mean_y.valueChanged.connect(self.resampling)
        self.sp_mean_y.setRange(-130, 130)
        # 控制高斯分布协方差矩阵的spinbox
        self.sp_conv_00 = QSpinBox()
        self.sp_conv_00.valueChanged.connect(self.resampling)
        self.sp_conv_00.setRange(-130, 130)
        self.sp_conv_01 = QSpinBox()
        self.sp_conv_01.valueChanged.connect(self.resampling)
        self.sp_conv_01.setRange(-130, 130)
        self.sp_conv_10 = QSpinBox()
        self.sp_conv_10.valueChanged.connect(self.resampling)
        self.sp_conv_10.setRange(-130, 130)
        self.sp_conv_11 = QSpinBox()
        self.sp_conv_11.valueChanged.connect(self.resampling)
        self.sp_conv_11.setRange(-130, 130)
        mean_layout = QHBoxLayout()
        mean_layout.addWidget(self.sp_mean_x)
        mean_layout.addWidget(self.sp_mean_y)
        conv_layout0 = QHBoxLayout()
        conv_layout0.addWidget(self.sp_conv_00)
        conv_layout0.addWidget(self.sp_conv_01)
        conv_layout1 = QHBoxLayout()
        conv_layout1.addWidget(self.sp_conv_10)
        conv_layout1.addWidget(self.sp_conv_11)
        conv_layout = QVBoxLayout()
        conv_layout.addLayout(conv_layout0)
        conv_layout.addLayout(conv_layout1)
        self.mean_groupbox = QGroupBox(title="均值")
        self.conv_groupbox = QGroupBox(title="协方差")
        self.mean_groupbox.setLayout(mean_layout)
        self.conv_groupbox.setLayout(conv_layout)
        # 样本数量控件
        self.sample_line_edit = QLineEdit()
        self.sample_line_edit.setValidator(QIntValidator(0, 10000))
        self.sample_line_edit.textChanged.connect(self.resampling)

        # 保存高斯分布参数的控件
        self.btn_save = QPushButton(self)
        self.btn_save.setText('保存')
        self.btn_save.clicked.connect(self.slot_btn_save)
        self.btn_pre_task = QPushButton(self)
        self.btn_pre_task.setText('前一任务')
        self.btn_pre_task.clicked.connect(self.slot_btn_pre)
        self.btn_next_task = QPushButton(self)
        self.btn_next_task.setText('下一任务')
        self.btn_next_task.clicked.connect(self.slot_btn_next)
        pre_next_btn_layout = QHBoxLayout()
        pre_next_btn_layout.addWidget(self.btn_pre_task)
        pre_next_btn_layout.addWidget(self.btn_next_task)
        self.btn_gen_planning_data = QPushButton(self)
        self.btn_gen_planning_data.setText('生成规划数据')
        self.btn_gen_planning_data.clicked.connect(self.generate_all_task_data)
        self.btn_test_planning_data = QPushButton(self)
        self.btn_test_planning_data.setText('测试规划数据')
        self.btn_test_planning_data.clicked.connect(self.test_planning_data)

        vLayout = QVBoxLayout()
        vLayout.addWidget(self.lb_task)
        vLayout.addWidget(self.mean_groupbox)
        vLayout.addWidget(self.conv_groupbox)
        vLayout.addWidget(self.sample_line_edit)
        vLayout.addWidget(self.btn_save)
        vLayout.addLayout(pre_next_btn_layout)
        vLayout.addWidget(self.btn_gen_planning_data)
        vLayout.addWidget(self.btn_test_planning_data)
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
        self.work_space = "/home/lys/catkin_ws/src/crawler_crane/crane_tutorials/crane_planning_data/"

        lift_obj_pos = [-33.0, 15.0, 20.0, -38.0, -9.0, 25.0, -4.0, 45.0, 20.0, -4.0, 30.0, 5.0,
                        -4.0, -7.0, 5.0, -14.0, -47.0, 10.0, -60.0, 98.0, 25.0, -60.0, 80.0, 10.0,
                        -40.0, 70.0, 10.0, -45.0, 45.0, 10.0, -45.0, 14.0, 10.0, -30.0, -47.0, 10.0,
                        -30.0, -42.0, 10.0, -41.0, -16.0, 5.0, -18.0, 8.0, 12.0, 8.0, 110.0, 10.0]
        #lift_obj_pos = [8.0, 110.0, 10.0, -40.0, -50.0, 15.0]
        lift_obj_pos = np.reshape(lift_obj_pos, (-1, 3))
        n = len(lift_obj_pos)
        self.planning_tasks = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.planning_tasks.append([list(lift_obj_pos[i]), list(lift_obj_pos[j])])
        self.planning_tasks = np.reshape(self.planning_tasks, (-1, 2, 3))
        self.tasks_num = len(self.planning_tasks)
        print(self.tasks_num)
        self.cur_task_id = 0
        self.cur_task = self.planning_tasks[self.cur_task_id]
        # 加载全局地图
        # self.global_map_ = np.loadtxt(self.work_space + "bin_map.txt")
        self.global_map_ = np.loadtxt("../LiftSceneRepresentation/LiftSceneData/bin_map.txt")

        # 加载已有的高斯分布参数
        params_file = self.work_space + "distribution_params.pkl"
        if (os.path.exists(params_file)):
            with open(params_file, "rb") as p_f:
                self.params_dict = pickle.load(p_f)
        else:
            self.params_dict = {}

        # # 测试
        # planning_data_file = self.work_space + "planning_data.pkl"
        # with open(planning_data_file, "rb") as p_f:
        #     self.planning_data = pickle.load(p_f)
        # self.task_keys = list(self.planning_data.keys())
        # print(self.task_keys)
        # self.cur_test_num = 0


    def load_task_data(self, task):
        # 加载局部地图
        obs_file = "({:d},{:d},{:d})".format(int(task[0][0]), int(task[0][1]),
                                             int(task[0][2]))
        self.local_map_init = np.loadtxt(self.work_space + "obs(人工挑选)/" + obs_file + "_local_map.txt")
        obs_file = "({:d},{:d},{:d})".format(int(task[1][0]), int(task[1][1]),
                                             int(task[1][2]))
        self.local_map_goal = np.loadtxt(self.work_space + "obs(人工挑选)/" + obs_file + "_local_map.txt")

        # 加载起重机器人无碰撞站位区
        file_name = self.work_space + "conf_free(人工挑选)/" + \
                    "({:d},{:d},{:d})_({:d},{:d},{:d})_conf_free".format(int(task[0][0]),
                                                                         int(task[0][1]),
                                                                         int(task[0][2]),
                                                                         int(task[1][0]),
                                                                         int(task[1][1]),
                                                                         int(task[1][2]))
        init_file_name = file_name + "_init.txt"
        goal_file_name = file_name + "_goal.txt"
        self.robot_potential_loc_init = np.loadtxt(init_file_name, delimiter=',', usecols=range(2))
        self.robot_potential_loc_goal = np.loadtxt(goal_file_name, delimiter=',', usecols=range(2))

        # 初始化高斯分布参数
        task_name = "({:d},{:d},{:d})_({:d},{:d},{:d})".format(int(task[0][0]),
                                                               int(task[0][1]),
                                                               int(task[0][2]),
                                                               int(task[1][0]),
                                                               int(task[1][1]),
                                                               int(task[1][2]))
        self.lb_task.setText("task_" + str(self.cur_task_id) + ":\n        " + task_name)
        if(task_name in self.params_dict.keys()):
            self.sample_num = self.params_dict[task_name]['sample_num']
            print(self.sample_num)
            self.sample_line_edit.setText(str(format(self.sample_num)))
            mean = self.params_dict[task_name]['mean']
            conv = self.params_dict[task_name]['conv']
            self.sp_mean_x.setValue(mean[0])
            self.sp_mean_y.setValue(mean[1])
            self.sp_conv_00.setValue(conv[0][0])
            self.sp_conv_01.setValue(conv[0][1])
            self.sp_conv_10.setValue(conv[1][0])
            self.sp_conv_11.setValue(conv[1][1])
        else:
            self.sample_line_edit.setText("5000")
            self.sp_mean_x.setValue(task[0][0])
            self.sp_mean_y.setValue(task[0][1])
            self.sp_conv_00.setValue(5.0)
            self.sp_conv_11.setValue(5.0)


    def visualize_global_map(self, color):
        rows = int(len(self.global_map_))
        x, y = np.nonzero(self.global_map_)
        x = x * self.scene_size_ / rows - 0.5 * self.scene_size_
        y = y * self.scene_size_ / rows - 0.5 * self.scene_size_
        self.ax.scatter(x, y, color=color, s=5, alpha=0.9)

    def visualize_local_map(self, lift_obj_pose, local_map, color):
        rows = int(len(local_map))
        x, y = np.nonzero(local_map)
        x = x * self.local_map_size_ / rows - 0.5 * self.local_map_size_ + lift_obj_pose[0]
        y = y * self.local_map_size_ / rows - 0.5 * self.local_map_size_ + lift_obj_pose[1]

        self.ax.scatter(x, y, color=color, s=5, alpha=0.3, cmap=plt.cm.hot_r)

    def visualize_robot_potential_location(self):
        self.ax.scatter(self.robot_potential_loc_init[:,0], self.robot_potential_loc_init[:,1], color='green', s=1, alpha=0.3)
        self.ax.scatter(self.robot_potential_loc_goal[:, 0], self.robot_potential_loc_goal[:, 1], color='red', s=1, alpha=0.3)

    def visualize_data(self):
        self.ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        # 绘制全局地图
        self.visualize_global_map('gray')

        # 绘制局部地图
        self.visualize_local_map(self.cur_task[0], self.local_map_init, 'yellow')
        self.visualize_local_map(self.cur_task[1], self.local_map_goal, 'purple')

        # 绘制起重机器人潜在站位区
        self.visualize_robot_potential_location()

        # 绘制高斯分布采样点
        self.ax.scatter(self.xytheta[:, 0], self.xytheta[:, 1], color='blue', s=1, alpha=0.8)


    def resampling(self):
        mean_x = self.sp_mean_x.value()
        mean_y = self.sp_mean_y.value()
        conv_00 = self.sp_conv_00.value()
        conv_01 = self.sp_conv_01.value()
        conv_10 = self.sp_conv_10.value()
        conv_11 = self.sp_conv_11.value()
        num = self.sample_line_edit.text()
        self.sample_num = int(num)
        #print(num)
        self.mean = np.array([mean_x, mean_y])  # 均值
        self.conv = np.array([[conv_00, conv_01],  # 协方差矩阵
                         [conv_10, conv_11]])
        xy = np.random.multivariate_normal(mean=self.mean, cov=self.conv, size=self.sample_num)
        theta = np.array(np.random.uniform(-3.1415926, 3.1415926, len(xy)))
        theta = theta[:, np.newaxis]
        #print(xy)
        self.xytheta = np.concatenate((xy, theta), axis=1)

        self.visualize_data()
        self.canvas.draw()  # 这里开始绘制

    def generate_all_task_data(self):
        n = len(self.planning_tasks)
        train_data = {}
        for i in range(n):
            task = self.planning_tasks[i]
            task_key = "task_{:d}".format(i)
            task_name = "({:d},{:d},{:d})_({:d},{:d},{:d})".format(int(task[0][0]),
                                                                   int(task[0][1]),
                                                                   int(task[0][2]),
                                                                   int(task[1][0]),
                                                                   int(task[1][1]),
                                                                   int(task[1][2]))
            #print(task_name)
            train_data[task_key] = {}
            if task_name in self.params_dict.keys():
                xytheta = self.generate_task_xytheta_by_sampling(self.params_dict[task_name])
                xytheta = clip(task[0], xytheta)  # 去掉不在起重机器人站位环中的点
                train_data[task_key]['xytheta'] = xytheta
            else:
                break

            train_data[task_key]['task_name'] = task_name
            train_data[task_key]['lift_obj_pose_init'] = task[0]
            train_data[task_key]['lift_obj_pose_goal'] = task[1]
            obs_file = "({:d},{:d},{:d})".format(int(task[0][0]), int(task[0][1]), int(task[0][2]))
            #print(obs_file)
            train_data[task_key]['obs_init'] = np.loadtxt(self.work_space + "obs/" + obs_file + "_local_gray_map.txt")
            #print(task_data['obs_init'])
            obs_file = "({:d},{:d},{:d})".format(int(task[1][0]), int(task[1][1]), int(task[1][2]))
            train_data[task_key]['obs_goal'] = np.loadtxt(self.work_space + "obs/" + obs_file + "_local_gray_map.txt")

            print(train_data[task_key])

        print(train_data)
        #planning_data_file = self.work_space + "planning_data.pkl"
        planning_data_file = self.work_space + "train_data1.pkl"
        with open(planning_data_file, "wb") as p_f:
            pickle.dump(train_data, p_f)

    def generate_task_xytheta_by_sampling(self, distribution_params):
        mean = distribution_params['mean']  # 均值
        conv = distribution_params['conv']  # 协方差矩阵
        num = distribution_params['sample_num']  # 样本数量
        xy = np.random.multivariate_normal(mean=mean, cov=conv, size=num)
        theta = np.array(np.random.uniform(-3.1415926, 3.1415926, len(xy)))
        theta = theta[:, np.newaxis]
        # print(xy)
        xytheta = np.concatenate((xy, theta), axis=1)
        return xytheta

    def test_planning_data(self):
        key = self.task_keys[self.cur_test_num]
        print(key)
        self.ax.cla()  # TODO:删除原图，让画布上只有新的一次的图
        # # 绘制全局地图
        self.visualize_global_map('gray')

        # 绘制局部地图
        lift_obj_pose_init = self.planning_data[key]['lift_obj_pose_init']
        lift_obj_pose_goal = self.planning_data[key]['lift_obj_pose_goal']
        local_map_init = self.planning_data[key]['obs_init']
        local_map_goal = self.planning_data[key]['obs_goal']
        self.visualize_local_map(lift_obj_pose_init, local_map_init, 'yellow')
        self.visualize_local_map(lift_obj_pose_goal, local_map_goal, 'purple')
        print(local_map_goal)

        # 绘制起重机器人潜在站位区
        file_name = self.work_space + "conf_free/" + self.planning_data[key]['task_name'] + "_conf_free"
        init_file_name = file_name + "_init.txt"
        goal_file_name = file_name + "_goal.txt"
        robot_potential_loc_init = np.loadtxt(init_file_name, delimiter=',', usecols=range(2))
        robot_potential_loc_goal = np.loadtxt(goal_file_name, delimiter=',', usecols=range(2))
        self.ax.scatter(robot_potential_loc_init[:, 0], robot_potential_loc_init[:, 1], color='green',
                        s=1, alpha=0.3)
        self.ax.scatter(robot_potential_loc_goal[:, 0], robot_potential_loc_goal[:, 1], color='red', s=1,
                        alpha=0.3)

        # 绘制高斯分布采样点
        xytheta = self.planning_data[key]['xytheta']
        self.ax.scatter(xytheta[:, 0], xytheta[:, 1], color='blue', s=1, alpha=0.8)

        self.canvas.draw()  # 这里开始绘制
        self.cur_test_num += 1


    def slot_btn_pre(self):
        try:

            if(self.cur_task_id -1 >= 0):
                self.cur_task_id -= 1
                self.cur_task = self.planning_tasks[self.cur_task_id]
                print("切换到任务：", self.cur_task)
                self.load_task_data(self.cur_task)
                self.visualize_data()
                self.canvas.draw()  # 这里开始绘制
            else:
                print("已是第0个任务了！！")

        except Exception as e:
            print(e)


    def slot_btn_next(self):
        try:
            if(self.cur_task_id + 1 < self.tasks_num):
                self.cur_task_id += 1
                self.cur_task = self.planning_tasks[self.cur_task_id]
                print("切换到任务：", self.cur_task)
                self.load_task_data(self.cur_task)
                self.visualize_data()
                self.canvas.draw()  # 这里开始绘制
            else:
                print("已是最后一个任务了！！")

        except Exception as e:
            print(e)


    def slot_btn_save(self):
        try:
            task_name = "({:d},{:d},{:d})_({:d},{:d},{:d})".format(int(self.cur_task[0][0]),
                                                                   int(self.cur_task[0][1]),
                                                                   int(self.cur_task[0][2]),
                                                                   int(self.cur_task[1][0]),
                                                                   int(self.cur_task[1][1]),
                                                                   int(self.cur_task[1][2]))
            params_file = self.work_space + "distribution_params.pkl"
            self.params_dict[task_name] = {}
            self.params_dict[task_name]['mean'] = self.mean
            self.params_dict[task_name]['conv'] = self.conv
            self.params_dict[task_name]['sample_num'] = self.sample_num
            with open(params_file, "wb") as params_f:
                pickle.dump(self.params_dict, params_f)

        except Exception as e:
            print("高斯分布参数保存时出现异常！")
            print(e)


def ui_main():
    app = QApplication(sys.argv)
    w = QtDraw()
    w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    ui_main()