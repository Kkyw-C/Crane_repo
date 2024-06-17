import xlrd
import numpy as np
import matplotlib.pyplot as plt

class LiftSceneRepresentationConverter:
    def __init__(self, obs_file, scene_size_x = 250.0, scene_size_y = 250.0, nx = 84, ny = 84, scene_org_x = 0.0, scene_org_y = 0.0):
        self.obs_file = obs_file
        self.scene_size_x = scene_size_x
        self.scene_size_y = scene_size_y
        self.nx = nx
        self.ny = ny
        self.scene_org_x = scene_org_x
        self.scene_org_y = scene_org_y
        self.obs_boxes = []
        self.obs_cylinders = []
        self.h_max = 0.0

    def generateSceneFile(self, scene_file):
        data = xlrd.open_workbook(self.obs_file)  # 打开xlsx文件
        scene_file += "_{:d}x{:d}.scene".format(self.nx, self.ny)
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


    def generateBinAndGrayMap(self, out_file):
        # 读取excel
        self.read_excel()

        step_x = self.scene_size_x / self.nx
        step_y = self.scene_size_y / self.ny

        # 创建二维数组
        bin_map = np.zeros((self.nx, self.ny), dtype=np.int)
        gray_map = np.zeros((self.nx, self.ny), dtype=np.int)

        # 每个小格分别与障碍物进行判断，看是否重合相交
        for i in range(self.nx):
            for j in range(self.ny):
                grid_x_min = i*step_x - 0.5 * self.scene_size_x + self.scene_org_x
                grid_x_max = (i+1)*step_x - 0.5 * self.scene_size_x + self.scene_org_x
                grid_y_min = j * step_y - 0.5 * self.scene_size_y + self.scene_org_y
                grid_y_max = (j+1) * step_y - 0.5 * self.scene_size_y + self.scene_org_y
                # 与所有长方体进行相交检测
                box_num = len(self.obs_boxes)  # 长方体个数
                for k in range(box_num):                                          # self.obs_boxes[k] = [长,宽,高,x,y,z,theta]

                    obs_x_min = self.obs_boxes[k][3] - 0.5 * self.obs_boxes[k][0]
                    obs_x_max = self.obs_boxes[k][3] + 0.5 * self.obs_boxes[k][0]
                    obs_y_min = self.obs_boxes[k][4] - 0.5 * self.obs_boxes[k][1]
                    obs_y_max = self.obs_boxes[k][4] + 0.5 * self.obs_boxes[k][1]
                    if(grid_x_min < obs_x_max and grid_x_max > obs_x_min and grid_y_min < obs_y_max and grid_y_max > obs_y_min):
                        bin_map[i][j] = 1
                        gray_map[i][j] = 255*self.obs_boxes[k][2] / self.h_max

                # 与所有圆柱体进行相交检测
                grid_org = np.array([0.5*(grid_x_min+grid_x_max), 0.5*(grid_y_min+grid_y_max)])   # 栅格中心坐标
                grid_right_top = grid_org + np.array([0.5*step_x, 0.5*step_y])
                a = grid_right_top - grid_org
                cylinder_num = len(self.obs_cylinders)  # 长方体个数
                for k in range(cylinder_num):                                 # self.obs_boxes[k] = [半径,高,x,y,z]
                    b = np.array([np.fabs(self.obs_cylinders[k][2] - grid_org[0]), np.fabs(self.obs_cylinders[k][3] - grid_org[1])])
                    c = b - a
                    if(c[0] < 0):
                        c[0] = 0
                    if(c[1] < 0):
                        c[1] = 0

                    if(np.linalg.norm(c) < self.obs_cylinders[k][0]):
                        bin_map[i][j] = 1
                        gray_map[i][j] = 255 * self.obs_boxes[k][2] / self.h_max

                    # obs_x_min = self.obs_cylinders[k][2] - self.obs_cylinders[k][0]
                    # obs_x_max = self.obs_cylinders[k][2] + self.obs_cylinders[k][0]
                    # obs_y_min = self.obs_cylinders[k][3] - self.obs_cylinders[k][0]
                    # obs_y_min = self.obs_cylinders[k][3] + self.obs_cylinders[k][0]

        # 存储数据
        c0_file_name = out_file + "/bin_map" + "_{:d}x{:d}.txt".format(self.nx, self.ny)
        c1_file_name = out_file + "/gray_map" + "_{:d}x{:d}.txt".format(self.nx, self.ny)
        print(bin_map.shape)
        self.text_save(c0_file_name, bin_map)
        self.text_save(c1_file_name, gray_map)

    def generateLocalBinAndGrayMap(self, lift_obj_pos, map_size, num, out_file):
        # 读取excel
        self.read_excel()

        step_x = step_y = map_size / num

        # 创建dict
        bin_map_dict = {}
        gray_map_dict = {}


        for obj_idx in range(len(lift_obj_pos)):
            # 创建二维数组
            bin_map = np.zeros((num, num), dtype=np.int)
            gray_map = np.zeros((num, num), dtype=np.int)
            # 每个小格分别与障碍物进行判断，看是否重合相交
            for i in range(num):
                for j in range(num):
                    grid_x_min = i * step_x - 0.5 * map_size + lift_obj_pos[obj_idx][0]
                    grid_x_max = (i + 1) * step_x - 0.5 * map_size + lift_obj_pos[obj_idx][0]
                    grid_y_min = j * step_y - 0.5 * map_size + lift_obj_pos[obj_idx][1]
                    grid_y_max = (j + 1) * step_y - 0.5 * map_size + lift_obj_pos[obj_idx][1]
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


            #plt.show()
            # 添加到dict中
            key = "({:d},{:d},{:d})_local_bin_map.txt".format(int(lift_obj_pos[obj_idx][0]), int(lift_obj_pos[obj_idx][1]), int(lift_obj_pos[obj_idx][2]))
            # np.savetxt(out_file + "/" + key, bin_map, fmt="%d")
            key1 = "({:d},{:d},{:d})_local_gray_map.txt".format(int(lift_obj_pos[obj_idx][0]), int(lift_obj_pos[obj_idx][1]),
                                                          int(lift_obj_pos[obj_idx][2]))
            np.savetxt(out_file + "/" + key, bin_map, fmt="%d")
            np.savetxt(out_file + "/" + key1, gray_map, fmt="%d")
            bin_map_dict[key] = bin_map
            gray_map_dict[key] = gray_map

        # 存储数据
        bin_map_file_name = out_file + "/local_bin_map_dict" + "_{:d}x{:d}".format(num, num)
        gray_map_file_name = out_file + "/local_gray_map_dict" + "_{:d}x{:d}".format(num, num)
        #print(bin_map_dict)
        np.save(bin_map_file_name, bin_map_dict)
        np.save(gray_map_file_name, gray_map_dict)


    def read_excel(self):
        workbook = xlrd.open_workbook(self.obs_file)  # 获取所有sheet
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


    # 把数据存入txt中
    def text_save(self, filename, data):
        # file = open(filename, 'w')
        # for i in range(len(data)):
        #     s = str(data[i]).replace('[', '').replace(']', '')  # 去除[],这两行按数据不同，可以选择
        #     s = s.replace("'", '').replace(',', '') + '\n'  # 去除单引号，逗号，每行末尾追加换行符
        #     file.write(s)
        # file.close()
        np.savetxt(filename, data, fmt='%d')
        print("保存文件成功\n")

def show_data():
    scene_size = 250.0
    plt.axes().set_aspect('equal')  # 为了绘制的图形不变形
    plt.title('Lifting Scene')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim([-0.5 * scene_size, 0.5 * scene_size])  # 设置x轴的边界
    plt.ylim([-0.5 * scene_size, 0.5 * scene_size])  # 设置y轴的边界


if __name__ == '__main__':
    lsrc = LiftSceneRepresentationConverter("./kyw_data/obs_kyw.xlsx")
    lsrc.generateSceneFile("./kyw_data/obs")
    lsrc.generateBinAndGrayMap("./kyw_data/")
    # lift_obj_pos = [-33.0, 15.0, 20.0, -38.0, -9.0, 25.0, -4.0, 45.0, 20.0, -4.0, 30.0, 5.0,
    #                 -4.0, -7.0, 5.0, -14.0, -47.0, 10.0, -60.0, 98.0, 25.0, -60.0, 80.0, 10.0,
    #                 -40.0, 70.0, 10.0, -45.0, 45.0, 10.0, -45.0, 14.0, 10.0, -30.0, -47.0, 10.0,
    #                 -30.0, -42.0, 10.0, -41.0, -16.0, 5.0, -18.0, 8.0, 12.0, 8.0, 110.0, 10.0,
    #                 -40.0, -50.0, 15.0]
    # lift_obj_pos = np.reshape(lift_obj_pos, (-1, 3))
    # lift_obj_pos_file = "/media/robotai/disk1/kyw/LiftSceneRepresentation/kyw_data/lift_obj_poses.txt"
    # lift_obj_pos = np.loadtxt(lift_obj_pos_file, delimiter=',')
    # lsrc.generateLocalBinAndGrayMap(lift_obj_pos, 56, 56, "/media/robotai/disk1/kyw/LiftSceneRepresentation/kyw_data/obs")
