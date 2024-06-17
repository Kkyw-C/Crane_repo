import numpy as np

data_file = "/home/lys/catkin_ws/src/crawler_crane/crane_tutorials/crane_planning_data/conf_free/lift_obj_poses.txt"
data = np.loadtxt(data_file, delimiter=',')
print(data)