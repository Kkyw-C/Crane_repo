import numpy as np
import matplotlib.pyplot as plt
import pickle

class PlanningDataVisualTool(object):
    def __init__(self):
        self.local_map_size_ = 56
        self.local_map_n_ = 56
        self.scene_size_ = 250  # 场景正方形的大小(m)
        self.scene_n_ = 250     # 场景栅格数量
        self.work_space = "/home/lys/catkin_ws/src/crawler_crane/crane_tutorials/crane_planning_data/"
        self.cur_task_name = ""

        # lift_obj_pos = [-33.0, 15.0, 20.0, -38.0, -9.0, 25.0, -4.0, 45.0, 20.0, -4.0, 30.0, 5.0,
        #                 -4.0, -7.0, 5.0, -14.0, -47.0, 10.0, -60.0, 98.0, 25.0, -60.0, 80.0, 10.0,
        #                 -40.0, 70.0, 10.0, -45.0, 45.0, 10.0, -45.0, 14.0, 10.0, -30.0, -47.0, 10.0,
        #                 -30.0, -42.0, 10.0, -41.0, -16.0, 5.0, -18.0, 8.0, 12.0, 8.0, 110.0, 10.0]
        lift_obj_pos = [-33.0, 15.0, 20.0, -4.0, 45.0, 20.0, -60.0, 80.0, 10.0]
        lift_obj_pos = np.reshape(lift_obj_pos, (-1, 3))
        n = len(lift_obj_pos)
        self.planning_tasks = []
        for i in range(n):
            for j in range(n):
                if i != j:
                    self.planning_tasks.append([list(lift_obj_pos[i]), list(lift_obj_pos[j])])
        self.planning_tasks = np.reshape(self.planning_tasks, (-1, 2, 3))
        #print(len(self.planning_tasks))

        # 加载全局地图
        self.global_map_ = np.loadtxt("../LiftSceneRepresentation/LiftSceneData/bin_map.txt")

        plt.axes().set_aspect('equal')  # 为了绘制的图形不变形
        plt.title('Lifting Scene')
        plt.xlabel('X')
        plt.ylabel('Y')

    def visualize_global_map(self, color):
        rows = int(len(self.global_map_))
        x, y = np.nonzero(self.global_map_)
        x = x * self.scene_size_ / rows - 0.5 * self.scene_size_
        y = y * self.scene_size_ / rows - 0.5 * self.scene_size_
        plt.scatter(x, y, color=color, s=1, alpha=0.3)

    def visualize_local_map(self, lift_obj_pose, local_map_file, color):
        local_map = np.loadtxt(local_map_file)
        rows = int(len(local_map))
        x, y = np.nonzero(local_map)
        x = x * self.local_map_size_ / rows - 0.5 * self.local_map_size_ + lift_obj_pose[0]
        y = y * self.local_map_size_ / rows - 0.5 * self.local_map_size_ + lift_obj_pose[1]
        plt.scatter(x, y, color=color, s=1, alpha=0.3)

    def visualize_local_map(self, lift_obj_pose, local_map, color):
        rows = int(len(local_map))
        x, y = np.nonzero(local_map)
        x = x * self.local_map_size_ / rows - 0.5 * self.local_map_size_ + lift_obj_pose[0]
        y = y * self.local_map_size_ / rows - 0.5 * self.local_map_size_ + lift_obj_pose[1]
        plt.scatter(x, y, color=color, s=1, alpha=0.3)

    def visualize_confs_distribution(self, lift_obj_pose_init, lift_obj_pose_goal):
        file_name = self.work_space + "conf_free/" + \
                    "({:d},{:d},{:d})_({:d},{:d},{:d})_conf_free".format(int(lift_obj_pose_init[0]), int(lift_obj_pose_init[1]),
                    int(lift_obj_pose_init[2]), int(lift_obj_pose_goal[0]), int(lift_obj_pose_goal[1]), int(lift_obj_pose_goal[2]))
        init_file_name = file_name + "_init.txt"
        goal_file_name = file_name + "_goal.txt"
        print(init_file_name)

        init_confs = np.loadtxt(init_file_name, delimiter=',', usecols=range(2))
        #print(init_confs)
        goal_confs = np.loadtxt(goal_file_name, delimiter=',', usecols=range(2))

        plt.scatter(init_confs[:,0], init_confs[:,1], color='green', s=1, alpha=0.3)
        plt.scatter(goal_confs[:, 0], goal_confs[:, 1], color='red', s=1, alpha=0.3)

    def generate_confs(self, lift_obj_pose_init, lift_obj_pose_goal):
        dir = self.work_space + "xytheta/"
        task_name = "({:d},{:d},{:d})_({:d},{:d},{:d})".format(int(lift_obj_pose_init[0]),
                                                               int(lift_obj_pose_init[1]),
                                                               int(lift_obj_pose_init[2]),
                                                               int(lift_obj_pose_goal[0]),
                                                               int(lift_obj_pose_goal[1]),
                                                               int(lift_obj_pose_goal[2]))
        # 反向吊装任务
        task_name1 = "({:d},{:d},{:d})_({:d},{:d},{:d})".format(int(lift_obj_pose_goal[0]),
                                                                int(lift_obj_pose_goal[1]),
                                                                int(lift_obj_pose_goal[2]),
                                                                int(lift_obj_pose_init[0]),
                                                                int(lift_obj_pose_init[1]),
                                                                int(lift_obj_pose_init[2]))
        file_name = dir + task_name + "_xytheta_init.txt"
        file_name1 = dir + task_name1 + "_xytheta_init.txt"


        # 确定正向任务起重机器人的x、y坐标
        num = 5000
        mean = np.array([-23.00, 55.00])  # 均值
        conv = np.array([[7.00, 8.00],  # 协方差矩阵
                         [0.00, 7.00]])
        xy = np.random.multivariate_normal(mean=mean, cov=conv, size=num)
        xy = self.clip(lift_obj_pose_init, xy)
        plt.scatter(xy[:, 0], xy[:, 1], color='gray', s=1, alpha=0.8)
        # 确定起重机人下车朝向theta[-pi, pi]
        theta = np.array(np.random.uniform(-3.1415926, 3.1415926, len(xy)))
        theta = theta[:, np.newaxis]
        xytheta = np.concatenate((xy, theta), axis=1)
        #print(xytheta)
        # 存储样本值
        # samples_dict = {}
        # samples_dict['task_name'] = task_name
        # samples_dict['lift_obj_pose_init'] = lift_obj_pose_init
        # samples_dict['lift_obj_pose_goal'] = lift_obj_pose_goal
        # samples_dict['xytheta_init'] = xytheta
        # with open(file_name + ".pkl", "wb") as sample_f:
        #     pickle.dump(samples_dict, sample_f)
        np.savetxt(file_name, xytheta, fmt="%.2f")
        # 存储高斯采样参数
        params = [lift_obj_pose_init + lift_obj_pose_goal + list(mean) + list(conv.flatten())]
        #print(params)
        np.savetxt(dir + task_name + "_sample_params_init.txt", params, fmt="%.2f")

        mean = np.array([-38.00, 73.00])  # 均值
        conv = np.array([[3.00, -5.00],  # 协方差矩阵
                         [6.00, 10.00]])
        xy = np.random.multivariate_normal(mean=mean, cov=conv, size=num)
        #xy = self.clip(lift_obj_pose_goal, xy)
        plt.scatter(xy[:, 0], xy[:, 1], color='gray', s=1, alpha=0.8)
        # 确定起重机人下车朝向theta[-pi, pi]
        theta = np.array(np.random.uniform(-3.1415926, 3.1415926, len(xy)))
        theta = theta[:, np.newaxis]
        xytheta = np.concatenate((xy, theta), axis=1)
        # print(xytheta)
        # 存储样本值
        # samples_dict = {}
        # samples_dict['task_name'] = task_name1
        # samples_dict['lift_obj_pose_init'] = lift_obj_pose_goal
        # samples_dict['lift_obj_pose_goal'] = lift_obj_pose_init
        # samples_dict['xytheta_init'] = xytheta
        # with open(file_name1 + ".pkl", "wb") as sample_f:
        #     pickle.dump(samples_dict, sample_f)
        np.savetxt(file_name1, xytheta, fmt="%.2f")
        # 存储高斯采样参数
        params = [lift_obj_pose_goal + lift_obj_pose_init + list(mean) + list(conv.flatten())]
        # print(params)
        np.savetxt(dir + task_name1 + "_sample_params_init.txt", params, fmt="%.2f")


    def clip(self, lift_obj_pose, data):
        #print(data)
        #print(len(data))
        confs = []
        count = 0
        R_min = 10
        R_max = 28
        for i in range(len(data)):
            d = np.sqrt((data[i][0] - lift_obj_pose[0]) ** 2 + (data[i][1] - lift_obj_pose[1]) ** 2)
            if R_min < d and d < R_max:
                confs.append(data[i])
        confs = np.array(confs)
        return confs

    def generate_data(self):
        n = len(self.planning_tasks)
        train_data = {}
        task_data = {}
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
            train_data[task_key]['task_name'] = task_name
            train_data[task_key]['lift_obj_pose_init'] = task[0]
            train_data[task_key]['lift_obj_pose_goal'] = task[1]
            obs_file = "({:d},{:d},{:d})".format(int(task[0][0]), int(task[0][1]), int(task[0][2]))
            #print(obs_file)
            train_data[task_key]['obs_init'] = np.loadtxt(self.work_space + "obs/" + obs_file + "_local_map.txt")
            #print(task_data['obs_init'])
            obs_file = "({:d},{:d},{:d})".format(int(task[1][0]), int(task[1][1]), int(task[1][2]))
            train_data[task_key]['obs_goal'] = np.loadtxt(self.work_space + "obs/" + obs_file + "_local_map.txt")
            train_data[task_key]['xytheta'] = np.loadtxt(self.work_space + "xytheta/" + task_name + "_xytheta_init.txt")
            #print(train_data[key])

        print(train_data)
        train_data_file = self.work_space + "train_data.pkl"
        with open(train_data_file, "wb") as train_f:
            pickle.dump(train_data, train_f)

    def display(self):
        train_data_file = self.work_space + "train_data.pkl"
        with open(train_data_file, "rb") as train_f:
            train_data = pickle.load(train_f)
        print(train_data)
        n = len(train_data)
        for i in range(0, n, 1):
            plt.axes().set_aspect('equal')  # 为了绘制的图形不变形
            plt.title('Lifting Scene')
            plt.xlabel('X')
            plt.ylabel('Y')

            task_key = "task_{:d}".format(i)
            task_data = train_data[task_key]
            self.visualize_global_map('gray')
            self.visualize_local_map(task_data['lift_obj_pose_init'], task_data['obs_init'], 'green')
            self.visualize_local_map(task_data['lift_obj_pose_goal'], task_data['obs_goal'], 'red')
            plt.scatter(task_data['xytheta'][:, 0], task_data['xytheta'][:, 1], color='gray')
            plt.show()





vt = PlanningDataVisualTool()
#vt.visualize_global_map('gray')
#vt.visualize_local_map([-60.0, 80.0, 10.0], "./LiftSceneData/(-60,80,10)_local_map.txt", 'green')
#vt.visualize_confs_distribution([-33.0, 15.0, 20.0], [-4.0, 45.0, 20.0])
#vt.generate_confs([-33.0, 15.0, 20.0], [-4.0, 45.0, 20.0])
# vt.visualize_confs_distribution([-33.0, 15.0, 20.0], [-60.0, 80.0, 10.0])
# vt.generate_confs([-33.0, 15.0, 20.0], [-60.0, 80.0, 10.0])
# vt.visualize_confs_distribution([-4.0, 45.0, 20.0], [-60.0, 80.0, 10.0])
# vt.generate_confs([-4.0, 45.0, 20.0], [-60.0, 80.0, 10.0])
#vt.generate_data()
vt.display()

plt.show()