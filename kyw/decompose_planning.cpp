#include <ros/ros.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/planning_scene_monitor/planning_scene_monitor.h> 
#include <fstream> 
#include <moveit/robot_state/conversions.h>
#include <moveit_msgs/DisplayTrajectory.h>
#include <moveit/robot_model/joint_model.h>
#include <moveit_visual_tools/moveit_visual_tools.h>
// #include <moveit/move_group_interface/move_group.h>
// #include <moveit/move_group_interface/move_group_interface.h>
// #include "crane_location_regions.h"
#include "msl/prm.h"
#include "msl/problem.h"
#include <visualization_msgs/Marker.h>
#include "msl/BiMRRTs.h"
#include "construct_robot_trajectory.h"

#include <torch/script.h> // One-stop header.


const double PI = 3.1415926;

void printVector(std::string str, const vector<double>& v)
{
    cout << str;
    unsigned int n = v.size();
    for(int i = 0; i < n; i++)
        cout << v[i] << "  ";
    cout << endl;
}

void split(const std::string& src, const std::string& separator, std::vector<std::string>& dest) //字符串分割到数组
{
    //参数1：要分割的字符串；参数2：作为分隔符的字符；参数3：存放分割后的字符串的vector向量
    std::string str = src;
    std::string substring;
    std::string::size_type start = 0, index;
    dest.clear();
    index = str.find_first_of(separator,start);
    do
    {
        if (index != std::string::npos)
        {
            substring = str.substr(start,index-start );
            dest.push_back(substring);
            start =index+separator.size();
            index = str.find(separator,start);
            if (start == std::string::npos) break;
        }
    }while(index != std::string::npos);

    //the last part
    substring = str.substr(start);
    dest.push_back(substring);
}

class MotionPlanningForCrane
{
public:
  MotionPlanningForCrane()
  {
    init();
  }
  ~MotionPlanningForCrane();
  void init();
  bool planBySamplingSingleConfig(std::string plannerName);
  bool planBySamplingSingleConfig(std::string plannerName, robot_state::RobotState& init_rs, robot_state::RobotState& goal_rs);
  void samplingSingleConfigTest(const std::vector<std::string>& planners, int nRuns);

  bool planByDecomposing(std::string basePlannerName, std::string upperPlannerName);
  bool samplingValidInitGoalConf(CraneLocationRegions& clr, robot_state::RobotState& init_rs, robot_state::RobotState& goal_rs);
  void decomposingTest(const std::vector<std::string>& basePlanners, const std::vector<std::string>& upperPlanners, int nRuns);

  bool isBaseWithDefaultUpperValid(const std::vector<double>& baseConf, robot_state::RobotState& rs);
  bool isCraneConfigValid(const std::vector<double>& craneConf, robot_state::RobotState& rs);
  bool planBasedPRM(std::string upperPlannerName, double dAllowedPlanningTime);

  void testSampleFromQMap();
  void testSamplebyIK();
  void testSampleFromCLR();
  double testplanBySamplingSingleConfig(std::string plannerName,int nRuns);
  void testplanBySamplingSingleConfigRandom(std::string plannerName,int nRuns);
  void testplanBySamplingSingleConfigCLR(std::string plannerName,int nRuns);

protected:
  bool generateValidStateByIK(const geometry_msgs::Pose& pose, planning_scene::PlanningScenePtr scene, robot_state::RobotState& rs);
  double computePathLength(const moveit::planning_interface::MoveGroupInterface::Plan& plan, double* dJointWeights);

    bool loadGlobalMap(string global_map_file, unsigned int** global_map);
    bool extractLocalMap(const geometry_msgs::Pose& pose, unsigned int **loc_map, bool bin);
    bool generateQMap();
    bool sampleFromQMap(bool for_init, std::vector<double> &crane_conf, robot_state::RobotState& rs);
    
//protected:
public:
  // 被吊物的起始位姿及终止位姿，只用了4个自由度
  geometry_msgs::Pose initPose_, goalPose_;
  moveit::planning_interface::MoveGroupInterface* whole_group_;
  const robot_state::JointModelGroup* joint_model_group_;
  planning_scene::PlanningScenePtr scene_;
  std::string resultPath_;
  robot_state::RobotStatePtr rs_;
  robot_trajectory::RobotTrajectoryPtr robot_traj_;

    string data_path;
    // 深度学习获得的分布模型
    torch::jit::script::Module distribution_module;

    // 存储全局地图，从磁盘读取
    unsigned int global_map_n;
    unsigned int **global_bin_map;
    unsigned int **global_gray_map;
    // 存储当前局部地图，从磁盘读取
    unsigned int local_map_size_;
    unsigned int local_map_n_;
    unsigned int **init_local_map_;
    unsigned int **goal_local_map_;
  // 站位质量图
    unsigned int **init_loc_quality_map_;
    unsigned int **goal_loc_quality_map_;
    double **init_loc_quality_ratio_;
    double **goal_loc_quality_ratio_;

    mutable random_numbers::RandomNumberGenerator rng_;
    CraneLocationRegions *init_clr_, *goal_clr_;

  // 调试用
  ros::Publisher jointPub_;
  moveit_msgs::RobotState rs_msg_;
  ros::Publisher display_publisher_;
  moveit_msgs::DisplayTrajectory display_trajectory_;
  // For visualizing things in rviz
  moveit_visual_tools::MoveItVisualToolsPtr visual_tools_;
  visualization_msgs::Marker markerVertexes_;
  void display(const geometry_msgs::Pose& liftObj_pose, unsigned int** loc_map);
  void visualize_data();

  // 统计时间
  timeval timer_;
};

class Vector2
{
public:
    double x, y;
    Vector2(double dX = 0.0, double dY = 0.0)
    {
        x = dX;
        y = dY;
    }
    ~Vector2(){};
    static Vector2 Max(Vector2 &v0, Vector2 &v1)
    {
        return Vector2(max(v0.x, v0.y), max(v1.x, v1.y));
    }
    double SqrMagnitude()
    {
        return x*x + y*y;
    }
};

bool isInterSection(double x_min, double x_max, double y_min, double y_max, double org_x, double org_y, double r, double R)
{
    double center_x = 0.5 * (x_min + x_max);
    double center_y = 0.5 * (y_min + y_max);
    double sqr_dist = (center_x - org_x) * (center_x - org_x) + (center_y - org_y) * (center_y - org_y);
    return  r * r < sqr_dist && sqr_dist < R * R;
}

MotionPlanningForCrane::~MotionPlanningForCrane()
{
  if(whole_group_)
    delete whole_group_;
}

void MotionPlanningForCrane::display(const geometry_msgs::Pose& liftObj_pose, unsigned int** loc_map)
{
    // 调试
    double x_min, y_min, step;
    step = local_map_size_ / local_map_n_;
    for(int i = 0; i < local_map_n_; i++)
    {
        for(int j = 0; j < local_map_n_; j++)
        {
            //printf("%d  ", this->init_local_map_[i][j]);
            if(loc_map[i][j])
            {
                geometry_msgs::Point vertex_;
                x_min = i * step -0.5 * local_map_size_ + liftObj_pose.position.x;
                y_min = j * step -0.5 * local_map_size_ + liftObj_pose.position.y;
                vertex_.x = x_min + 0.5 * step;
                vertex_.y = y_min + 0.5 * step;
                //cout << vertex_.x << ", " << vertex_.y << endl;
                markerVertexes_.points.push_back(vertex_);
            }
        }
        //printf("\n");
    }
    
    int k = 0;
    while(k < 2000)
    {
        visual_tools_->publishMarker(markerVertexes_);
        visual_tools_->trigger();
        k++;
    }

}

void MotionPlanningForCrane::visualize_data()
{
  // 显示被吊物位置
  visual_tools_->publishSphere(initPose_, rviz_visual_tools::colors::RED);
  // 显示qmap的真值

  // 显示qmap的预测值
}

void MotionPlanningForCrane::init()
{
    // 设置被吊物起始位姿和终止位姿
    /*initPose_.position.x = -26.3;
    initPose_.position.y = -47.20;
    initPose_.position.z = 7.0;
    initPose_.orientation.w = 1;
    goalPose_.position.x = 99.85;
    goalPose_.position.y = -31.9;
    goalPose_.position.z = 9.52;
    goalPose_.orientation.w = 1;*/
    initPose_.position.x = -47;//8.0;
    initPose_.position.y = -33;//110.0;
    initPose_.position.z = 13;//10.0;
    initPose_.orientation.w = 1;
    goalPose_.position.x = -18.0;
    goalPose_.position.y = 8.0;
    goalPose_.position.z = 12;
    goalPose_.orientation.w = 1;

    init_clr_ = new CraneLocationRegions(initPose_, 8.0, 28.0);
    goal_clr_ = new CraneLocationRegions(goalPose_, 8.0, 28.0);

    data_path = std::string("/home/kieran/LearnSampleDistribution/crane_planning__data/");
    int i, j;
    global_map_n = 250;
    global_bin_map = new unsigned int* [global_map_n];
    global_gray_map = new unsigned int* [global_map_n];
    for(i = 0; i < global_map_n; i++)
    {
        global_bin_map[i] = new unsigned int [global_map_n];
        global_gray_map[i] = new unsigned int [global_map_n];
    }
    // 从磁盘加载全局地图
    string global_bin_map_file = data_path + "global_bin_map.txt";
    string global_gray_map_file = data_path + "global_gray_map.txt";
    loadGlobalMap(global_bin_map_file, global_bin_map);
    loadGlobalMap(global_gray_map_file, global_gray_map);

    try {
        // Deserialize the ScriptModule from a file using torch::jit::load().
        distribution_module = torch::jit::load("/home/kieran/LearnSampleDistribution/th__models/traced_model.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
    }

    local_map_size_ = 56;
    local_map_n_ = 56;

    // 为相关数组分配内存空间
    init_local_map_ = new unsigned int* [local_map_n_];
    goal_local_map_ = new unsigned int* [local_map_n_];
    init_loc_quality_map_ = new unsigned int* [local_map_n_];
    goal_loc_quality_map_ = new unsigned int* [local_map_n_];
    init_loc_quality_ratio_ = new double* [local_map_n_];
    goal_loc_quality_ratio_ = new double* [local_map_n_];

    for(i = 0; i < local_map_n_; i++)
    {
        init_local_map_[i] = new unsigned int [local_map_n_];
        goal_local_map_[i] = new unsigned int [local_map_n_];
        init_loc_quality_map_[i] = new unsigned int [local_map_n_];
        goal_loc_quality_map_[i] = new unsigned int [local_map_n_];
        init_loc_quality_ratio_[i] = new double [local_map_n_];
        goal_loc_quality_ratio_[i] = new double [local_map_n_];
    }

    // 获取起吊区和就位区的局部地图
    extractLocalMap(initPose_, init_local_map_, false);
    extractLocalMap(goalPose_, goal_local_map_, false);
    // 通过基于深度学习的站位分布模型获取无碰撞的站位质量图
    generateQMap();




    // 设置规划组及相关配置工作
    whole_group_ = new moveit::planning_interface::MoveGroupInterface("whole");
    robot_state::RobotState robot_state_ = robot_state::RobotState(*whole_group_->getCurrentState());
    joint_model_group_ = robot_state_.getJointModelGroup("whole");


    // std::vector<const robot_model::JointModel::Bounds*> limits = joint_model_group_->getActiveJointModelsBounds();
    // for(int i = 0; i < limits.size(); i++)
    // {
    //   ROS_ERROR("%d: [%f, %f]", i, limits[i]->at(0).min_position_, limits[i]->at(0).max_position_);
    // }
    // joint_model_group_ = robot_state_.getJointModelGroup("upper");
    // int n = joint_model_group_->getVariableCount() - joint_model_group_->getMimicJointModels().size();
    // ROS_ERROR("NUM = %d", n);
    // joint_model_group_ = robot_state_.getJointModelGroup("base");
    // n = joint_model_group_->getVariableCount()- joint_model_group_->getMimicJointModels().size();
    // ROS_ERROR("NUM = %d", n);

    // 获得当前活动的场景，为碰撞检测做好准备
    planning_scene_monitor::PlanningSceneMonitorPtr monitor_ptr_udef;
    monitor_ptr_udef.reset(new planning_scene_monitor::PlanningSceneMonitor("robot_description"));
    monitor_ptr_udef->requestPlanningSceneState("get_planning_scene");
    planning_scene_monitor::LockedPlanningSceneRW ps(monitor_ptr_udef);
    ps->getCurrentStateNonConst().update();
    scene_ = ps->diff();
    scene_->decoupleParent();

    // 设置结果存储路径
    resultPath_ = ros::package::getPath("crane_tutorials") + std::string("/motion__planning_results_kyw");
    
    rs_ = whole_group_->getCurrentState();
    // 调试用
    ros::NodeHandle node_handle;
    jointPub_ = node_handle.advertise<sensor_msgs::JointState>("joint_states", 1);
    display_publisher_ = node_handle.advertise<moveit_msgs::DisplayTrajectory>("/move_group/display_planned_path", 1, true);

    visual_tools_.reset(new moveit_visual_tools::MoveItVisualTools("odom","/moveit_visual_markers"));
    visual_tools_->deleteAllMarkers();
    markerVertexes_.header.frame_id  = "odom";
    markerVertexes_.header.stamp = ros::Time::now();
    markerVertexes_.action = visualization_msgs::Marker::ADD;
    markerVertexes_.pose.orientation.w = 1.0;
    markerVertexes_.id = 0;
    markerVertexes_.type = visualization_msgs::Marker::POINTS;
    markerVertexes_.scale.x = 0.5;
    markerVertexes_.scale.y = 0.5;
    markerVertexes_.color.g = 1.0f;
    markerVertexes_.color.a = 1.0;
}


bool MotionPlanningForCrane::loadGlobalMap(string global_map_file, unsigned int** global_map)
{
    std::ifstream fin(global_map_file);
    int i = 0, j = 0;
    if(fin) {
        cout << "fin is loading success" << endl;
        std::string str;
        std::vector<std::string> data;
        while (getline(fin, str)) {
            //ROS_INFO("%s", str.c_str());
            split(str, " ", data);
            for(j = 0; j < global_map_n; j++)
            {
                global_map[i][j] = atoi(data[j].c_str());
            }
            i++;
        }
        return true;
    }
    else
    {
        return false;
    }
}


bool MotionPlanningForCrane::extractLocalMap(const geometry_msgs::Pose& pose, unsigned int **loc_map, bool bin)
{
    int org_x_idx, org_y_idx, left_idx, bottom_idx;
    int i, j;
    org_x_idx = int(pose.position.x+125);
    org_y_idx = int(pose.position.y+125);
    left_idx = org_x_idx - int(0.5*local_map_n_);
    bottom_idx = org_y_idx - int(0.5*local_map_n_);
    unsigned int** global_map = bin ? global_bin_map : global_gray_map;
    for(i = 0; i < local_map_n_; i++)
    {
        for(j = 0; j < local_map_n_; j++)
        {
            if((0 <= left_idx+i && left_idx+i < global_map_n) && (0 <= bottom_idx+j && bottom_idx+j < global_map_n))  // 判断该单元格是否出了全局地图
            {
                loc_map[i][j] = global_map[left_idx + i][bottom_idx + j];
            }
            else
            {
                loc_map[i][j] = bin ? 1 : 256;
            }
        }
    }

    return true;
}

bool MotionPlanningForCrane::generateQMap()
{
    torch::Tensor init_obs = torch::zeros({1, 56*56});
    torch::Tensor goal_obs = torch::zeros({1, 56*56});
    for(int i = 0; i < 56; i++)
    {
        for(int j = 0; j < 56; j++)
        {
            init_obs[0][56*i+j] = (int)init_local_map_[i][j];
            goal_obs[0][56*i+j] = (int)goal_local_map_[i][j];
        }
    }

    std::vector<torch::jit::IValue> init_inputs;
    std::vector<torch::jit::IValue> goal_inputs;
    init_inputs.emplace_back(init_obs);
    goal_inputs.emplace_back(goal_obs);

    // Execute the model and turn its output into a tensor.
    at::Tensor init_output = distribution_module.forward(init_inputs).toTensor();
    at::Tensor goal_output = distribution_module.forward(goal_inputs).toTensor();

    int init_q, goal_q;
    double init_q_sum, goal_q_sum;
    init_q_sum = goal_q_sum = 0.0;
    for(int i = 0; i < 56; i++)
    {
        for(int j = 0; j < 56; j++)
        {
            init_q = init_output[0][56*i+j].item().toInt();
            goal_q = goal_output[0][56*i+j].item().toInt();
            init_loc_quality_map_[i][j] = init_q > 0 ? init_q : 0;
            goal_loc_quality_map_[i][j] = goal_q > 0 ? goal_q : 0;
            //cout << init_loc_quality_map_[i][j] << "  ";
            init_q_sum += init_loc_quality_map_[i][j];
            goal_q_sum += goal_loc_quality_map_[i][j];
        }
        //cout << endl;
    }

    // 计算各单元格百分比
    for(int i = 0; i < 56; i++)
    {
        for(int j = 0; j < 56; j++)
        {
            init_loc_quality_ratio_[i][j] = init_loc_quality_map_[i][j] / init_q_sum;
            goal_loc_quality_ratio_[i][j] = goal_loc_quality_map_[i][j] / goal_q_sum;
        }
    }
}

bool MotionPlanningForCrane::sampleFromQMap(bool for_init, std::vector<double> &crane_conf, robot_state::RobotState& rs)
{
  //display(initPose_, init_loc_quality_map_);
    unsigned int **qmap = for_init ? init_loc_quality_map_ : goal_loc_quality_map_;
    double **qmap_ratio = for_init ? init_loc_quality_ratio_ : goal_loc_quality_ratio_;
    geometry_msgs::Pose liftObj_Pose_ = for_init ? initPose_ : goalPose_;
    CraneLocationRegions clr(liftObj_Pose_, 10.0, 28.0);
    
    // 选择单元格
    double p = rng_.uniformReal( 0.0, 1.0 );
    //int i = 0, j = 0;
    int sel_i, sel_j;
    sel_i = sel_j = 0;
    double sum = 0.0;
    for(int n = 0; n < 56*56; n++)
    {
      sum += qmap_ratio[n/56][n%56];
      //cout << n/56 << "  " << n%56 << endl;
      if( sum >= p )
      {
        sel_i = n/56;
        sel_j = n%56;
        break;
      }	    
    }

    // 选中单元格sel_i,sel_j
    //cout << "p = " << p << ", Selected cell (" << sel_i << ", " << sel_j << ")" << endl;
    std::vector<double> base_conf(3, 0.0);
    double step_x, step_y;
    step_x = step_y = local_map_size_ / local_map_n_;
    base_conf[0] = rng_.uniformReal(sel_i * step_x, (sel_i + 1) * step_x);
    base_conf[1] = rng_.uniformReal(sel_j * step_y, (sel_j + 1) * step_y);
    base_conf[0] += -0.5 * local_map_size_ + liftObj_Pose_.position.x;
    base_conf[1] += -0.5 * local_map_size_ + liftObj_Pose_.position.y;
    base_conf[2] = rng_.uniformReal(-PI, PI);
    clr.upperIK(base_conf, crane_conf);
    //printVector("base_conf: ", base_conf);
    
    if( isCraneConfigValid(crane_conf, rs) )
    {
        //rs.printStatePositions();
        return true;
    }

    return false;
}

void MotionPlanningForCrane::testSampleFromQMap()
{
    gettimeofday(&timer_, NULL);
  double dStartTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);

    robot_state::RobotState rs(*whole_group_->getCurrentState());
    std::vector<geometry_msgs::Point> vis_points;
    geometry_msgs::Point pt;
    int nMaxSamples = 10000;
    int nValid = 0;
    std::vector<double> crane_conf(8, 0.0);
    for(int i = 0; i < nMaxSamples; i++)
    {
        bool bValid = sampleFromQMap(true, crane_conf, rs);

        //// 同步到rviz中，用于调试
       moveit_msgs::RobotState rs_msg;
       robot_state::robotStateToRobotStateMsg(rs, rs_msg);
       jointPub_.publish(rs_msg.joint_state);
      // // ROS_INFO("%f, %f, %f", pose.position.x, pose.position.y, pose.position.z);
       rs.printStatePositions();
       sleep(0.5);

        if(bValid)
        {
           nValid += 1;
          //   ROS_INFO("i = %d, Valid = %d", i, nValid);
           pt.x = crane_conf[0];
          pt.y = crane_conf[1];
          pt.z = 0.0;
          vis_points.push_back(pt);
          visual_tools_->publishSpheres(vis_points, rviz_visual_tools::colors::ORANGE, 0.5, "confs");
          visual_tools_->trigger();
        }
        // ROS_INFO("i = %d, Valid = %d", i, nValid);
      //  else{
      //    pt.x = crane_conf[0];
      //   pt.y = crane_conf[1];
      //   pt.z = 0.0;
      //   vis_points.push_back(pt);
      //   visual_tools_->publishSpheres(vis_points, rviz_visual_tools::colors::BLUE, 0.5, "confs");
      //   visual_tools_->trigger();
      //  }
        // pt.x = crane_conf[0];
        // pt.y = crane_conf[1];
        // pt.z = 0.0;
        // vis_points.push_back(pt);
        // visual_tools_->publishSpheres(vis_points, rviz_visual_tools::colors::RED, 0.5, "confs");
        // visual_tools_->trigger();
        
    }
    sleep(0.2);

    gettimeofday(&timer_, NULL);
  double dCurrentTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
  double t = dCurrentTime - dStartTime;

  cout << "sampleFromQMap: t = " << t << ", nAttempt = " << nMaxSamples << ", nValid = " << nValid << endl;
}

void MotionPlanningForCrane::testSamplebyIK()
{
  gettimeofday(&timer_, NULL);
  double dStartTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
    std::vector<geometry_msgs::Point> vis_points;
    geometry_msgs::Point pt;
  robot_state::RobotState rs(*whole_group_->getCurrentState());
  int nMaxSamples = 10000;
  int nValidSamples = 0;

  int j = 0;
    while (j < nMaxSamples) {
        //ROS_INFO("j: %d", j);
        // 求逆解
        bool bIK = rs.setFromIK(joint_model_group_, initPose_, 2, 0.5);

        // // 判断是否碰撞
        if (bIK && !scene_->isStateColliding(rs, "whole")) {
          moveit_msgs::RobotState rs_msg;
          robot_state::robotStateToRobotStateMsg(rs, rs_msg);
          jointPub_.publish(rs_msg.joint_state);
            nValidSamples += 1;
            double *joint_pos = rs.getVariablePositions();   // 0, 1, 2, 5, 6, 8, 9对应起重机位形
            // 输出无碰撞位形
            // for(int k = 0; k < 10; k++)
            // {
            //     if( k != 3 && k != 4 && k != 7 )
            //         init_out << joint_pos[k] << ",";
            // }
            // init_out << endl;
            pt.x = joint_pos[0];
            pt.y = joint_pos[1];
            pt.z = 0.0;
            vis_points.push_back(pt);
            visual_tools_->publishSpheres(vis_points, rviz_visual_tools::colors::YELLOW, 0.45, "confs");
            visual_tools_->trigger();
        }
        j++;
    }
    sleep(0.2);
    gettimeofday(&timer_, NULL);
  double dCurrentTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
  double t = dCurrentTime - dStartTime;

  cout << "SamplebyIK: t = " << t << ", nAttempt = " << nMaxSamples << ", nValid = " << nValidSamples << endl;
}

void MotionPlanningForCrane::testSampleFromCLR()
{
  CraneLocationRegions clr(initPose_, 10.0, 28.0);
  gettimeofday(&timer_, NULL);
  double dStartTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
  std::vector<geometry_msgs::Point> vis_points;
  geometry_msgs::Point pt;
  robot_state::RobotState rs(*whole_group_->getCurrentState());
    int nMaxSamples = 10000;
    int nValid = 0;
    std::vector<double> crane_conf(8, 0.0);
    for(int i = 0; i < nMaxSamples; i++)
    {
        bool bValid = clr.directSampling(crane_conf);

        
       //ROS_INFO("%f, %f, %f", pose.position.x, pose.position.y, pose.position.z);
      //  rs.printStatePositions();
      //  sleep(2);

        if( isCraneConfigValid(crane_conf, rs) )
        {
          //// 同步到rviz中，用于调试
          moveit_msgs::RobotState rs_msg;
          robot_state::robotStateToRobotStateMsg(rs, rs_msg);
          jointPub_.publish(rs_msg.joint_state);
          rs.printStatePositions();
          pt.x = crane_conf[0];
          pt.y = crane_conf[1];
          pt.z = 0.0;
          vis_points.push_back(pt);
          visual_tools_->publishSpheres(vis_points, rviz_visual_tools::colors::BLUE, 0.5, "confs");
          visual_tools_->trigger();
          nValid += 1;
        }
        //ROS_INFO("i = %d, Valid = %d", i, nValid);
    }

    gettimeofday(&timer_, NULL);
  double dCurrentTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
  double t = dCurrentTime - dStartTime;

  cout << "SampleFromCLR: t = " << t << ", nAttempt = " << nMaxSamples << ", nValid = " << nValid << endl;
}

bool MotionPlanningForCrane:: planBySamplingSingleConfig(std::string plannerName)
{
  double dStartTime, dCurrentTime, dSamplingInitConfTime, dSamplingGoalConfTime, dPlanningTime;
  bool bInitFound, bGoalFound, bPathFound;
  bInitFound = bGoalFound = bPathFound = false;

  // 起重机器人起始/终止状态采样
  robot_state::RobotState init_rs(*whole_group_->getCurrentState());
  robot_state::RobotState goal_rs(*whole_group_->getCurrentState());
  gettimeofday(&timer_, NULL);
  dStartTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
  bInitFound = generateValidStateByIK(initPose_, scene_, init_rs);
  if(bInitFound)
    ROS_INFO("Initial Robot State Found!!");
  else
    ROS_INFO("Initial Robot State NOT Found!!");

  gettimeofday(&timer_, NULL);
  dCurrentTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
  dSamplingInitConfTime = dCurrentTime - dStartTime;

  dStartTime = dCurrentTime;
  bGoalFound = generateValidStateByIK(goalPose_, scene_, goal_rs);
  if(bGoalFound)
    ROS_INFO("Goal Robot State Found!!");
  else
    ROS_INFO("Goal Robot State NOT Found!!");
  gettimeofday(&timer_, NULL);
  dCurrentTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
  dSamplingGoalConfTime = dCurrentTime - dStartTime;

  // 设置规划任务
  moveit::planning_interface::MoveGroupInterface::Plan whole_plan;   // 其中的joint_trajectory只包含所规划的几个自由度，不包括mimic joint
  double dMaxPlanningTime = 120;
  // if(bInitFound && bGoalFound)
  // {
    // 设置工作空间范围
    whole_group_->setWorkspace(-80, -130, -1, 80, 130, 100);
    whole_group_->setPlannerId(plannerName);
    whole_group_->setPlanningTime(dMaxPlanningTime);
    whole_group_->setNumPlanningAttempts(2);
    // 设置起始位形
    whole_group_->setStartState(init_rs);
    // 设置目标位形
    whole_group_->setJointValueTarget(goal_rs);
    //whole_group_->setPoseTarget(goalPose_);  // 无法规划，规划失败！！！
    // 求解规划问题
    bPathFound = whole_group_->plan(whole_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS;
    if(bPathFound)
    {
      ROS_INFO("FOUND A PATH FOR CRANE!!");
      dPlanningTime = whole_plan.planning_time_;
    }
    else
    {
      ROS_INFO("FAILURE!!");
      dPlanningTime = dMaxPlanningTime;
    }
  // }
  // else
  // {
  //   dPlanningTime = 0.0;
  // }

  // 对结果进行后处理，记录：起吊位形，就位位形，起始位形采样时间，终止位形采样时间，规划是否成功，规划时间，路径长度
  // world_joint/x, world_joint/y, world_joint/theta, chassis_to_wheel_left_joint, chassis_to_wheel_right_joint,
  // superStructure_joint, boom_joint, rope_joint, hook_block_joint, hook_joint
  double *init_pos = init_rs.getVariablePositions();
  double *goal_pos = goal_rs.getVariablePositions();
  double dJointWeights[7] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  double dPathLen;
  if(bPathFound)
    dPathLen = computePathLength(whole_plan, dJointWeights);
  else
    dPathLen = 10e-40;
  std::string strFileName = resultPath_ + "/" + plannerName + ".csv";
	std::ofstream fout( strFileName.c_str(), std::ios::out|std::ios::app );
  static bool bFirstWrite = true;
	if( bFirstWrite )
	{
		fout << "起吊位形x" << ",起吊位形y" << ",起吊位形alpha" << ",起吊位形beta" << ",起吊位形gama" << ",起吊位形h" << ",起吊位形w"
		     << ",就位位形x" << ",就位位形y" << ",就位位形alpha" << ",就位位形beta" << ",就位位形gama" << ",就位位形h" << ",就位位形w"
		     << ",找到路径" << ",路径长度" << ",总规划时间" << ",起始位形采样成功" << ",起始位形采样时间"
		     << ",目标位形采样成功" << ",目标位形采样时间" << ",规划算法运行时间";
		bFirstWrite = false;
	}
	fout.seekp( 0, std::ios::end );

	fout << "\n" << init_pos[0] << ","  << init_pos[1] << ","  << init_pos[2] << ","  << init_pos[5] << ","  << init_pos[6] << ","  << init_pos[8] << ","  << init_pos[9] << ","
            << goal_pos[0] << ","  << goal_pos[1] << ","  << goal_pos[2] << ","  << goal_pos[5] << ","  << goal_pos[6] << ","  << goal_pos[8] << ","  << goal_pos[9] << ","
            << bPathFound << "," << dPathLen << "," << dSamplingInitConfTime + dSamplingGoalConfTime + dPlanningTime << ","
         << bInitFound << "," << dSamplingInitConfTime << "," << bGoalFound << "," << dSamplingGoalConfTime << "," << dPlanningTime;
	fout.close();

	return bPathFound;
}

bool MotionPlanningForCrane::planBySamplingSingleConfig(std::string plannerName, robot_state::RobotState& init_rs, robot_state::RobotState& goal_rs)
{
    double dStartTime, dCurrentTime, dSamplingInitConfTime, dSamplingGoalConfTime, dPlanningTime;
    bool bInitFound, bGoalFound, bPathFound;
    bInitFound = bGoalFound = true;
    bPathFound = false;

    gettimeofday(&timer_, NULL);
    dStartTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);

    dSamplingInitConfTime = 0.0;
    dSamplingGoalConfTime = 0.0;

    // 设置规划任务
    moveit::planning_interface::MoveGroupInterface::Plan whole_plan;   // 其中的joint_trajectory只包含所规划的几个自由度，不包括mimic joint
    double dMaxPlanningTime = 120;

    // 设置工作空间范围
    whole_group_->setWorkspace(-120, -130, -1, 120, 130, 100);
    whole_group_->setPlannerId(plannerName);
    whole_group_->setPlanningTime(dMaxPlanningTime);
    whole_group_->setNumPlanningAttempts(2);
    // 设置起始位形
    whole_group_->setStartState(init_rs);
    // 设置目标位形
    whole_group_->setJointValueTarget(goal_rs);
    //whole_group_->setPoseTarget(goalPose_);  // 无法规划，规划失败！！！
    // 求解规划问题
    bPathFound = whole_group_->plan(whole_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS;
    if(bPathFound)
    {
        ROS_INFO("FOUND A PATH FOR CRANE!!");
        dPlanningTime = whole_plan.planning_time_;
    }
    else
    {
        ROS_INFO("FAILURE!!");
        dPlanningTime = dMaxPlanningTime;
    }

    // 对结果进行后处理，记录：起吊位形，就位位形，起始位形采样时间，终止位形采样时间，规划是否成功，规划时间，路径长度
    // world_joint/x, world_joint/y, world_joint/theta, chassis_to_wheel_left_joint, chassis_to_wheel_right_joint,
    // superStructure_joint, boom_joint, rope_joint, hook_block_joint, hook_joint
    double *init_pos = init_rs.getVariablePositions();
    double *goal_pos = goal_rs.getVariablePositions();
    double dJointWeights[7] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    double dPathLen;
    if(bPathFound)
        dPathLen = computePathLength(whole_plan, dJointWeights);
    else
        dPathLen = 10e-40;
    std::string strFileName = resultPath_ + "/" + plannerName + ".csv";
    // ROS_INFO("%s", strFileName.c_str());
    std::ofstream fout( strFileName.c_str(), std::ios::out|std::ios::app );
    static bool bFirstWrite = true;
    if( bFirstWrite )
    {
        fout << "起吊位形x" << ",起吊位形y" << ",起吊位形alpha" << ",起吊位形beta" << ",起吊位形gama" << ",起吊位形h" << ",起吊位形w"
             << ",就位位形x" << ",就位位形y" << ",就位位形alpha" << ",就位位形beta" << ",就位位形gama" << ",就位位形h" << ",就位位形w"
             << ",找到路径" << ",路径长度" << ",总规划时间" << ",起始位形采样成功" << ",起始位形采样时间"
             << ",目标位形采样成功" << ",目标位形采样时间" << ",规划算法运行时间";
        bFirstWrite = false;
    }
    fout.seekp( 0, std::ios::end );

    fout << "\n" << init_pos[0] << ","  << init_pos[1] << ","  << init_pos[2] << ","  << init_pos[5] << ","  << init_pos[6] << ","  << init_pos[8] << ","  << init_pos[9] << ","
         << goal_pos[0] << ","  << goal_pos[1] << ","  << goal_pos[2] << ","  << goal_pos[5] << ","  << goal_pos[6] << ","  << goal_pos[8] << ","  << goal_pos[9] << ","
         << bPathFound << "," << dPathLen << "," << dSamplingInitConfTime + dSamplingGoalConfTime + dPlanningTime << ","
         << bInitFound << "," << dSamplingInitConfTime << "," << bGoalFound << "," << dSamplingGoalConfTime << "," << dPlanningTime;
    fout.close();

    return bPathFound;
}

//自定义的调用采样器位形qmap的路径规划验证
double MotionPlanningForCrane::testplanBySamplingSingleConfig(std::string plannerName,int nRuns)
{
	int success_count=0,fail_count=0,fail_weixing=0;
  // int nRuns=100;
	//这两个的rs区别是在isCraneConfigValid函数中进行区分？
	robot_state::RobotState init_rs(*whole_group_->getCurrentState());
  robot_state::RobotState goal_rs(*whole_group_->getCurrentState());
	std::vector<double> crane_conf_init(8, 0.0);
	std::vector<double> crane_conf_goal(8, 0.0);
  gettimeofday(&timer_, NULL);
  double dStartTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
	//下面两句获得初始位形和目标位形
	for (int i=0;i<nRuns;i++)
	{
		bool initConf = sampleFromQMap(true, crane_conf_init, init_rs);
		bool goalConf = sampleFromQMap(false, crane_conf_goal, goal_rs);
		if(initConf && goalConf )
		{
			bool pvalid = planBySamplingSingleConfig(plannerName, init_rs, goal_rs);
			if(pvalid){
          success_count += 1;
          ROS_INFO("i = %d, Valid = %d", i, success_count);
      }
      else{fail_count+=1;}
		}
    else
    {
      fail_weixing +=1;
    }
	}
  double dEndTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
  double dTime = dEndTime-dStartTime;
	double success = (double)success_count / (nRuns-fail_weixing);
	cout << "count = " << success_count <<"  and fail count= "<<fail_count<< "   and success is :"<<success<<"     time is :   "<<dTime<<endl;
  cout << "fail weixing count  zidingyi=" <<fail_weixing<< endl;
  return success;
	
}

//IK方法的路径规划验证
void MotionPlanningForCrane::testplanBySamplingSingleConfigRandom(std::string plannerName,int nRuns)
{
  gettimeofday(&timer_, NULL);
  double dCurrentTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
	int success_count=0,fail_count=0;
	//下面两句获得初始位形和目标位形
	for (int i=0;i<nRuns;i++)
	{
			bool pvalid = planBySamplingSingleConfig(plannerName);
			if(pvalid){
          success_count += 1;
            ROS_INFO("i = %d, Valid = %d", i, success_count);
      }
      else{fail_count+=1;}
	}
  double dEndTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
  double dTime = dEndTime-dCurrentTime;
	double success = (double)success_count / nRuns;
	cout << "count = " << success_count <<"and fail count= "<<fail_count<< "   and success is :"<<success<<"    and time is   "<<dTime<<endl;
	
}

//CLR方法的路径规划验证
void MotionPlanningForCrane::testplanBySamplingSingleConfigCLR(std::string plannerName,int nRuns)
{
	int success_count=0,fail_count=0,fail_weixing=0;
	robot_state::RobotState init_rs(*whole_group_->getCurrentState());
  robot_state::RobotState goal_rs(*whole_group_->getCurrentState());
	std::vector<double> crane_conf_init(8, 0.0);
	std::vector<double> crane_conf_goal(8, 0.0);
  CraneLocationRegions init_clr(initPose_, 10.0, 28.0);
  CraneLocationRegions goal_clr(goalPose_, 10.0, 28.0);
  gettimeofday(&timer_, NULL);
  double dStartTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
	//下面两句获得初始位形和目标位形
	for (int i=0;i<nRuns;i++)
	{
    bool initValid = init_clr.directSampling(crane_conf_init);
		bool initConf =  isCraneConfigValid(crane_conf_init,init_rs);
    bool goalValid = goal_clr.directSampling(crane_conf_goal);
		bool goalConf = isCraneConfigValid(crane_conf_goal, goal_rs);
		if(initConf && goalConf )
		{
			bool pvalid = planBySamplingSingleConfig(plannerName, init_rs, goal_rs);
			if(pvalid){
          success_count += 1;
            ROS_INFO("i = %d, Valid = %d", i, success_count);
      }
      else{fail_count+=1;}
		}
    else
    {
      fail_weixing +=1;
    }
	}
  double dEndTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
  double dTime = dEndTime-dStartTime;
	double success = (double)success_count / (nRuns-fail_weixing);
	cout << "count = " << success_count <<"  and fail count= "<<fail_count<< "   and success is :"<<success<<"     time is :   "<<dTime<<endl;
  cout << "fail weixing count =" <<fail_weixing<< endl;
	
}

void MotionPlanningForCrane::samplingSingleConfigTest(const std::vector<std::string>& planners, int nRuns)
{
  int nPlanners = planners.size();
  for(int i = 0; i < nPlanners; i++)
  {
    ROS_INFO("Testing %s:", planners[i].c_str());
    for(int j = 0; j < nRuns; j++)
    {
      ROS_INFO("The %d th running", j);
      planBySamplingSingleConfig(planners[i]);
      // testplanBySamplingSingleConfig(planners[i],nRuns);
    }
  }
}

bool MotionPlanningForCrane::samplingValidInitGoalConf(CraneLocationRegions& clr, robot_state::RobotState& init_rs, robot_state::RobotState& goal_rs)
{
  bool bInitFound, bGoalFound;
  bInitFound = bGoalFound = false;
  robot_state::RobotState rs(*whole_group_->getCurrentState());   // 这是耗时操作，可将其放到构造函数
  std::vector<double> baseConf(3, 0.0);
  // 设置起重机器人行走过程中最安全的上车位形
  std::vector<double> craneConf(8, 0.0);
  craneConf[4] = 1.45;
  craneConf[5] = -1.45;
  craneConf[6] = 50;
  rs.setJointGroupPositions(joint_model_group_, craneConf);
  // 用最安全的上车位形来在站位扇环中进行采样，获得无碰撞的起始和目标位形
  while(ros::ok())
  {
    clr.samplingBaseConfig(baseConf);
    craneConf[0] = baseConf[0];
    craneConf[1] = baseConf[1];
    craneConf[2] = baseConf[2];
    craneConf[3] = 0.0;
    craneConf[4] = 1.45;
    craneConf[5] = -1.45;
    craneConf[6] = 50;
    craneConf[7] = 0.0;
    rs.setJointGroupPositions(joint_model_group_, craneConf);
    // // 同步到rviz中，用于调试
    // robot_state::robotStateToRobotStateMsg(rs, rs_msg_);
    // jointPub_.publish(rs_msg_.joint_state);
    if(!scene_->isStateColliding(rs, "whole"))
    {
      goal_rs = rs;
      bGoalFound = true;
      ROS_INFO("Goal Collision Free");
    }
    else
    {
      bGoalFound = false;
      ROS_ERROR("COLLISION");
    }
    //sleep(5);

    if(bGoalFound)
    {
      // 根据pick_goal_rs求解pick_init_rs
      clr.upperIK(baseConf, craneConf);
      rs.setJointGroupPositions(joint_model_group_, craneConf);
      // // 同步到rviz中，用于调试
      // robot_state::robotStateToRobotStateMsg(rs, rs_msg_);
      // jointPub_.publish(rs_msg_.joint_state);
      if(!scene_->isStateColliding(rs, "whole"))
      {
        init_rs = rs;
        bInitFound = true;
        ROS_INFO("Initial Collision Free");
        break;
      }
      else
      {
        bInitFound = false;
        ROS_ERROR("COLLISION");
      }
      //sleep(5);
    }
    ROS_INFO("\n");
  }

  return (bInitFound && bGoalFound);
}


bool MotionPlanningForCrane::planByDecomposing(std::string basePlannerName, std::string upperPlannerName)
{
  double dStartTime, dCurrentTime, dSamplingInitConfTime, dSamplingGoalConfTime, dPlanningTime;
  bool bInitFound, bGoalFound, bPathFound;
  bInitFound = bGoalFound = bPathFound = false;

  double dPickPlanningTime, dPlacePlanningTime, dMovePlanningTime;
  bool bPickSuccess, bPlaceSuccess, bMoveSuccess;
  bPickSuccess = bPlaceSuccess = bMoveSuccess = false;
  double dMaxPlanningTime = 10.0;

  // PICK PLANNING
  moveit::planning_interface::MoveGroupInterface upper_group("upper");
  moveit::planning_interface::MoveGroupInterface::Plan pick_plan;
  // 起重机器人起始/终止状态采样
  robot_state::RobotState pick_init_rs(*whole_group_->getCurrentState());   // 非常耗时的操作，慎用
  robot_state::RobotState pick_goal_rs(*whole_group_->getCurrentState());
  CraneLocationRegions pickCLR(initPose_);
  gettimeofday(&timer_, NULL);
  dStartTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
  bool bSamplingSuccess = samplingValidInitGoalConf(pickCLR, pick_init_rs, pick_goal_rs);
  gettimeofday(&timer_, NULL);
  dCurrentTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
  dSamplingInitConfTime = dCurrentTime - dStartTime;
  if( bSamplingSuccess )
  {
    bInitFound = true;
    // 起吊规划
    dMaxPlanningTime = 10;
    upper_group.setPlannerId(upperPlannerName);
    upper_group.setPlanningTime(dMaxPlanningTime);
    upper_group.setNumPlanningAttempts(2);
    // 设置起始位形
    upper_group.setStartState(pick_init_rs);
    // 设置目标位形
    upper_group.setJointValueTarget(pick_goal_rs);
    // 求解规划问题
    bPickSuccess = upper_group.plan(pick_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS;
    if(bPickSuccess)
    {
      ROS_INFO("PICKING SUCCESS!!");
      dPickPlanningTime = pick_plan.planning_time_;
    }
    else
    {
      ROS_INFO("FAILURE!!");
      dPickPlanningTime = dMaxPlanningTime;
    }
  }


  // PLACE PLANNING
  moveit::planning_interface::MoveGroupInterface::Plan place_plan;
  // 起重机器人起始/终止状态采样
  robot_state::RobotState place_init_rs(*whole_group_->getCurrentState());
  robot_state::RobotState place_goal_rs(*whole_group_->getCurrentState());
  CraneLocationRegions placeCLR(goalPose_);
  gettimeofday(&timer_, NULL);
  dStartTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
  bSamplingSuccess = samplingValidInitGoalConf(placeCLR, place_goal_rs, place_init_rs);
  gettimeofday(&timer_, NULL);
  dCurrentTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
  dSamplingGoalConfTime = dCurrentTime - dStartTime;
  if( bSamplingSuccess )
  {
    bGoalFound = true;
    // 起吊规划
    dMaxPlanningTime = 10;
    upper_group.setPlannerId(upperPlannerName);
    upper_group.setPlanningTime(dMaxPlanningTime);
    upper_group.setNumPlanningAttempts(2);
    // 设置起始位形
    upper_group.setStartState(place_init_rs);
    // 设置目标位形
    upper_group.setJointValueTarget(place_goal_rs);
    // 求解规划问题
    bPlaceSuccess = upper_group.plan(place_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS;
    if(bPlaceSuccess)
    {
      ROS_INFO("PLACING SUCCESS!!");
      dPlacePlanningTime = place_plan.planning_time_;
    }
    else
    {
      ROS_INFO("FAILURE!!");
      dPlacePlanningTime = dMaxPlanningTime;
    }
  }

  // MOVE PLANNING
  moveit::planning_interface::MoveGroupInterface base_group("base");
  moveit::planning_interface::MoveGroupInterface::Plan move_plan;
  dMaxPlanningTime = 80;
  // 设置工作空间范围
    base_group.setWorkspace(-60, -50, -1, 150, 40, 100);
    base_group.setPlannerId(basePlannerName);
    base_group.setPlanningTime(dMaxPlanningTime);
    base_group.setNumPlanningAttempts(2);
    // 设置起始位形
    base_group.setStartState(pick_goal_rs);
    // 设置目标位形
    base_group.setJointValueTarget(place_init_rs);
    // 求解规划问题
    bMoveSuccess = base_group.plan(move_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS;
    if(bMoveSuccess)
    {
      ROS_INFO("MOVE SUCCESS!!");
      dMovePlanningTime = move_plan.planning_time_;
    }
    else
    {
      ROS_INFO("FAILURE!!");
      dMovePlanningTime = dMaxPlanningTime;
    }

    display_trajectory_.trajectory_start = pick_plan.start_state_;
    if(bPickSuccess)
      display_trajectory_.trajectory.push_back(pick_plan.trajectory_);
    if(bMoveSuccess)
      display_trajectory_.trajectory.push_back(move_plan.trajectory_);
    if(bPlaceSuccess)
      display_trajectory_.trajectory.push_back(place_plan.trajectory_);
    display_publisher_.publish(display_trajectory_);


  // 对结果进行后处理，记录：起始位形采样时间，终止位形采样时间，规划是否成功，规划时间，路径长度
  double dBaseJointWeights[3] = {1.0, 1.0, 1.0};
  double dUpperJointWeights[4] = {1.0, 1.0, 1.0, 1.0};
  double dPathLen;
  bPathFound = bPickSuccess && bMoveSuccess && bPlaceSuccess;
  if(bPathFound)
    dPathLen = computePathLength(move_plan, dBaseJointWeights) + computePathLength(pick_plan, dUpperJointWeights) + computePathLength(place_plan, dUpperJointWeights);
  else
    dPathLen = 10e-40;
  std::string strFileName = resultPath_ + "/decompose/B_" + basePlannerName + "+U_" + upperPlannerName + ".csv";
	std::ofstream fout( strFileName.c_str(), std::ios::out|std::ios::app );
  static bool bFirstWrite = true;
	if( bFirstWrite )
	{
		fout << "找到路径" << ",路径长度" << ",总规划时间" << ",起始位形采样成功" << ",起始位形采样时间" << ",目标位形采样成功" << ",目标位形采样时间" << ",规划算法运行时间"
         << ",起吊规划时间" << ",行走规划时间" << ",就位规划时间";
		bFirstWrite = false;
	}
	fout.seekp( 0, std::ios::end );
  dPlanningTime = dPickPlanningTime + dMovePlanningTime + dPlacePlanningTime;
  double dTotalPlanningTime = dSamplingInitConfTime + dSamplingGoalConfTime + dPlanningTime;
  
	fout << "\n" << bPathFound << "," << dPathLen << "," << dTotalPlanningTime << "," 
       << bInitFound << "," << dSamplingInitConfTime << "," << bGoalFound << "," << dSamplingGoalConfTime << "," << dPlanningTime << ","
       << dPickPlanningTime << "," << dMovePlanningTime << "," << dPlacePlanningTime;
	fout.close();
}

void MotionPlanningForCrane::decomposingTest(const std::vector<std::string>& basePlanners, const std::vector<std::string>& upperPlanners, int nRuns)
{
  int nPlanners = basePlanners.size();
  for(int i = 0; i < nPlanners; i++)
  {
    ROS_INFO("Testing %s + %s :", basePlanners[i].c_str(), upperPlanners[i].c_str());
    for(int j = 0; j < nRuns; j++)
    {
      ROS_INFO("The %d th running", j);
      planByDecomposing(basePlanners[i], upperPlanners[i]);
    }
  }
}

bool MotionPlanningForCrane::isBaseWithDefaultUpperValid(const std::vector<double>& baseConf, robot_state::RobotState& rs)
{
  std::vector<double> craneConf(8, 0.0);
  craneConf[0] = baseConf[0];
  craneConf[1] = baseConf[1];
  craneConf[2] = baseConf[2];
  craneConf[3] = 0.0;
  craneConf[4] = 1.45;
  craneConf[5] = -1.45;
  craneConf[6] = 50;
  craneConf[7] = 0.0;
  rs.setJointGroupPositions(joint_model_group_, craneConf);
	scene_->setCurrentState(rs);
  robot_state::RobotState& current_state = scene_->getCurrentStateNonConst();
	bool bSatisfied = scene_->isStateValid(current_state, "whole");
  return bSatisfied;
}

bool MotionPlanningForCrane::isCraneConfigValid(const std::vector<double>& craneConf, robot_state::RobotState& rs)
{
  rs.setJointGroupPositions(joint_model_group_, craneConf);
	scene_->setCurrentState(rs);
  robot_state::RobotState& current_state = scene_->getCurrentStateNonConst();
	bool bSatisfied = scene_->isStateValid(current_state, "whole");
  return bSatisfied;
}

bool MotionPlanningForCrane::planBasedPRM(std::string upperPlannerName, double dAllowedPlanningTime)
{
  double dRemainingTime = dAllowedPlanningTime;
  gettimeofday(&timer_, NULL);
  double dStartTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
  bool bSuccess = false;
  moveit::planning_interface::MoveGroupInterface::Plan pick_plan, place_plan, move_plan;

  // 1.用最安全的上车位姿构建roadmap
  CraneLocationRegions pickCLR(initPose_), placeCLR(goalPose_);
  Problem* pProlem = new Problem("base", &pickCLR, &placeCLR);
  PRM prm(pProlem);
  prm.bVisual_ = true;
  ROS_INFO("KKKKKKKKKKKKKKKKKKKKKKKKK");
    
  // 构建roadmap
  prm.Construct(3000);

  // 2.利用CLR采样站位环内的下车位姿，并尝试规划，若经常失败，则增量继续构建roadmap，直到规划成功
  robot_state::RobotState pick_init_rs(*whole_group_->getCurrentState());
  robot_state::RobotState pick_goal_rs(*whole_group_->getCurrentState());
  robot_state::RobotState place_init_rs(*whole_group_->getCurrentState());
  robot_state::RobotState place_goal_rs(*whole_group_->getCurrentState());
  double dMovePlanningTime = 0;
  gettimeofday(&timer_, NULL);
  double dCurrentTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
  dRemainingTime = dAllowedPlanningTime - (dCurrentTime - dStartTime);
  int nAttempts = 20;  // 尝试规划10次，若均失败则需增加节点
  bool bMoveSuccess = false;
  std::vector<double> craneConf(8, 0.0);
  std::vector<double> v;
  double dMovePathLen = 0;
  while(!bMoveSuccess && dRemainingTime > 0.0)
  {
    if(nAttempts > 0)   // 尝试规划
    {
      nAttempts--;
      while(1)      // 确保：1）起点和终点有效，否则无法进行下车规划；2）对应的起吊位形和就位位形是有效的，否则两端上车规划无法规划。
      {
        pickCLR.samplingBaseConfig(v);
        pickCLR.upperIK(v, craneConf);
        isBaseWithDefaultUpperValid(v, pick_goal_rs);
        if(!isBaseWithDefaultUpperValid(v, pick_goal_rs) || !isCraneConfigValid(craneConf, pick_init_rs))
        {
          continue;
        }
        pProlem->InitialState = MSLVector(v[0], v[1], v[2]);

        placeCLR.samplingBaseConfig(v);
        placeCLR.upperIK(v, craneConf);
        if(!isBaseWithDefaultUpperValid(v, place_init_rs) || !isCraneConfigValid(craneConf, place_goal_rs))
        {
          continue;
        }
        pProlem->GoalState = MSLVector(v[0], v[1], v[2]);
    
        break;
      }

      // 寻找路径
      bMoveSuccess = prm.Plan();
      if(bMoveSuccess)    // 对规划结果进行处理
      {
        // 构造moveit风格的路径
        // 计算路径长度
        list<MSLVector>::iterator it;
        int i = 0;
        it=prm.Path.begin();
        MSLVector prePoint = (*it);
        it++;
        for(; it!=prm.Path.end(); it++)
        {
          double dSegLen = 0;
          int d = (*it).dim();
          for(int j = 0; j < d; j++)
          {
            dSegLen += ((*it)[j] - prePoint[j]) * ((*it)[j] - prePoint[j]);
          }
          dMovePathLen += sqrt(dSegLen); 
          prePoint = (*it);
        } 
      }

      ROS_INFO("nAttempts: %d", 20-nAttempts);
    }
    else        // 失败次数过多，需要补充roadmap节点
    {
      prm.Construct(500);
      nAttempts = 10;
    }

    gettimeofday(&timer_, NULL);
    dCurrentTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
    dRemainingTime = dAllowedPlanningTime - (dCurrentTime - dStartTime);
    ROS_INFO("dRemainingTime: %f", dRemainingTime);
  }

  dMovePlanningTime = dCurrentTime - dStartTime;
  cout << "MovePlanningTime: " << dMovePlanningTime << endl;

  // 3.规划两端的上车运动，即Picking planning和Placing planning
  bool bPickSuccess, bPlaceSuccess;
  bPickSuccess = bPlaceSuccess = false;
  if(bMoveSuccess)    // 下车规划成功，则规划两端的上车
  {
    moveit::planning_interface::MoveGroupInterface upper_group("upper");
    upper_group.setPlannerId(upperPlannerName);
    // 设置起吊端起始、终止位形, pick_init_rs和pick_goal_rs在上面已求得
    upper_group.setStartState(pick_init_rs);        // 设置起始位形
    upper_group.setJointValueTarget(pick_goal_rs);  // 设置目标位形
    // 设置就位端起始、终止位形, place_init_rs和place_goal_rs在上面已求得
    upper_group.setStartState(place_init_rs);          // 设置起始位形
    upper_group.setJointValueTarget(place_goal_rs);    // 设置目标位形

    // 求解规划问题
    while(dRemainingTime > 0.0 && (!bPickSuccess || !bPlaceSuccess))
    {
      // 起吊端规划
      if(!bPickSuccess)
      {
        upper_group.setPlanningTime(dRemainingTime);
        bPickSuccess = upper_group.plan(pick_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS;
        if(bPickSuccess)
        {
          ROS_INFO("PICKING SUCCESS!!");
        }
        else
        {
          ROS_INFO("FAILURE!!");
        }
      }
      gettimeofday(&timer_, NULL);
      dCurrentTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
      dRemainingTime = dAllowedPlanningTime - (dCurrentTime - dStartTime);

      // 就位端规划
      if(!bPlaceSuccess)
      {
        upper_group.setPlanningTime(dRemainingTime);
        bPlaceSuccess = upper_group.plan(place_plan) == moveit::planning_interface::MoveItErrorCode::SUCCESS;
        if(bPlaceSuccess)
        {
          ROS_INFO("PLACING SUCCESS!!");
        }
        else
        {
          ROS_INFO("FAILURE!!");
        }
      }
      gettimeofday(&timer_, NULL);
      dCurrentTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
      dRemainingTime = dAllowedPlanningTime - (dCurrentTime - dStartTime);
      ROS_INFO("dRemainingTime: %f", dRemainingTime);
    }
  }

  gettimeofday(&timer_, NULL);
  dCurrentTime = timer_.tv_sec+(timer_.tv_usec/1000000.0);
  double dTotalPlanningTime = dCurrentTime - dStartTime;

  bool bFinalSuccess = bMoveSuccess && bPickSuccess && bPlaceSuccess;
  double dUpperJointWeights[4] = {1.0, 1.0, 1.0, 1.0};
  double dPathLen;
  if(bFinalSuccess)
    dPathLen = dMovePathLen + computePathLength(pick_plan, dUpperJointWeights) + computePathLength(place_plan, dUpperJointWeights);
  else
    dPathLen = 10e-40;
  std::string strFileName = resultPath_ + "/decompose/PRMBased+" + upperPlannerName + ".csv";
	std::ofstream fout( strFileName.c_str(), std::ios::out|std::ios::app );
  static bool bFirstWrite = true;
	if( bFirstWrite )
	{
		fout << "找到路径" << ",路径长度" << ",总规划时间" << ",行走规划时间";
		bFirstWrite = false;
	}
	fout.seekp( 0, std::ios::end );  
	fout << "\n" << bFinalSuccess << "," << dPathLen << "," << dTotalPlanningTime << "," << dMovePlanningTime;
	fout.close();

  return bFinalSuccess;
}

bool MotionPlanningForCrane::generateValidStateByIK(const geometry_msgs::Pose& pose, planning_scene::PlanningScenePtr scene, robot_state::RobotState& rs)
{
  int nMaxSamples = 30000;
  bool bFound = false;
  int i = 0;
  //const robot_state::JointModelGroup* joint_model_group = scene->getCurrentState().getJointModelGroup("whole");
  while(i<nMaxSamples)
  {
    // 求逆解
    bool bIK = rs.setFromIK(joint_model_group_, pose, 2, 0.5);
    
    // 判断是否碰撞
    if(bIK && !scene->isStateColliding(rs, "whole"))
    {
       // 同步到rviz中，用于调试
       moveit_msgs::RobotState rs_msg;
       robot_state::robotStateToRobotStateMsg(rs, rs_msg_);
       jointPub_.publish(rs_msg_.joint_state);
      //  ROS_INFO("%f, %f, %f", pose.position.x, pose.position.y, pose.position.z);
      //  rs.printStatePositions();
      //  sleep(2);

      bFound = true;
      break;
    }
    i++;
  }

  return bFound;
}

double MotionPlanningForCrane::computePathLength(const moveit::planning_interface::MoveGroupInterface::Plan& plan, double* dJointWeights)
{
  double dLen = 0.0;
  trajectory_msgs::JointTrajectory joint_traj = plan.trajectory_.joint_trajectory;
  int nWaypoints = joint_traj.points.size();
  int nJoints = joint_traj.points[0].positions.size();
  //ROS_INFO("%d-------------------%d", nWaypoints, nJoints);
  
  double dSegLen;
  for(int i = 1; i < nWaypoints; i++)
  {
    dSegLen = 0.0;
    for(int j = 0; j < nJoints; j++)
    {
       dSegLen += dJointWeights[j] * pow((joint_traj.points[i].positions[j] - joint_traj.points[i-1].positions[j]), 2);
    }
    dLen += sqrt(dSegLen);
  }

  return dLen;
}



int main(int argc, char **argv)
{
  ros::init(argc, argv, "motion_planning_for_crane");
  ros::NodeHandle node_handle;  
  ros::AsyncSpinner spinner(1);
  spinner.start();

//   rviz_visual_tools::RvizVisualToolsPtr visual_tools_;
//   visual_tools_.reset(new rviz_visual_tools::RvizVisualTools("base_footprint","/AAAAAAAAAAAAA"));

// while(ros::ok())
// {
//   // Create pose
// Eigen::Affine3d pose, pose1;                                                                                                                                                                                                                                                                                                                                                                                                                                                  
// visual_tools_->generateRandomPose(pose);
// visual_tools_->generateRandomPose(pose1);
// // pose = Eigen::AngleAxisd(M_PI/4, Eigen::Vector3d::UnitY()); // rotate along X axis by 45 degrees
// // pose.translation() = Eigen::Vector3d( 0.1, 0.1, 0.1 ); // translate x,y,z

// // Publish arrow vector of pose
// ROS_INFO_STREAM_NAMED("test","Publishing Arrow");
// visual_tools_->publishCuboid(visual_tools_->convertPoseToPoint(pose), visual_tools_->convertPoseToPoint(pose1), rviz_visual_tools::RED);
// sleep(1);
// }


//    MotionPlanningForCrane mpc;
//    std::vector<std::string> planners;
//   planners.push_back("RRTConnectkConfigDefault");
//   planners.push_back("SBLkConfigDefault");
//   planners.push_back("ESTkConfigDefault");
//   planners.push_back("LBKPIECEkConfigDefault");
//   planners.push_back("BKPIECEkConfigDefault");
//   planners.push_back("KPIECEkConfigDefault");
//   planners.push_back("RRTkConfigDefault");
//   planners.push_back("RRTstarkConfigDefault");
//   // //planners.push_back("TRRTkConfigDefault");   //似乎无法正常运行，move_group崩溃，需要设置额外的参数
//   planners.push_back("PRMkConfigDefault");
//   planners.push_back("PRMstarkConfigDefault");
////   planners.push_back("CBiMRRTConfigDefault");
//   mpc.samplingSingleConfigTest(planners, 50000);
    // moveit_visual_tools::MoveItVisualToolsPtr visual_tools_;
    MotionPlanningForCrane mpc;
    // mpc.initPose_.position.x = 100;//-23;
    // mpc.initPose_.position.y = 55;//65;
    // mpc.initPose_.position.z = 10;//9;
    // mpc.initPose_.position.x = 100;//8.0;
    // mpc.initPose_.position.y = 55;//110.0;
    // mpc.initPose_.position.z = 10;//10.0;
    //最终位置点2
    // mpc.goalPose_.position.x = 100;//8.0;
    // mpc.goalPose_.position.y = 55;//110.0;
    // mpc.goalPose_.position.z = 10;//10.0;
    // mpc.initPose_.position.x = 100;//8.0;
    // mpc.initPose_.position.y = 55;//110.0;
    // mpc.initPose_.position.z = 10;//10.0;
    // mpc.goalPose_.position.x = 30;//8.0;
    // mpc.goalPose_.position.y = 26;//110.0;
    // mpc.goalPose_.position.z = 5;//10.0;
    // mpc.initPose_.position.x = -38;//8.0;
    // mpc.initPose_.position.y = 105;//110.0;
    // mpc.initPose_.position.z = 5;//10.0;
    //最终位置点1
    mpc.initPose_.position.x = -38;//8.0;
    mpc.initPose_.position.y = 105;//110.0;
    mpc.initPose_.position.z = 5;//10.0;
    //如何将起重机位置绘制出来？
    // mpc.initPose_.orientation.w = 1;
    // goalPose_.position.x = -18.0;
    // goalPose_.position.y = 8.0;
    // goalPose_.position.z = 12;
    // goalPose_.orientation.w = 1;
    // std::vector<std::string> planners;
    // planners.push_back("RRTConnectkConfigDefault");
    // planners.push_back("PRMkConfigDefault");
    // planners.push_back("RRTkConfigDefault");
    // mpc.samplingSingleConfigTest(planners, 100);

//测试CLR成功率？
    // double suc[1000]={0};
    // double sum = 0;
    // for(int i = 0;i<1000;i++){
    //   suc[i]=mpc.testplanBySamplingSingleConfig("RRTConnect",1000);
    //   sum += suc[i];
    //   ROS_INFO("i=%d",i);
    // }
    // cout << "success is :   "<< sum/1000 << endl;

    // mpc.testplanBySamplingSingleConfig("RRTConnect",1000);
    // mpc.testplanBySamplingSingleConfigRandom("TRRT",1000);
    // mpc.testplanBySamplingSingleConfigCLR("RRT",1000);
    //用于绘制
    mpc.testSampleFromQMap();
    mpc.testSamplebyIK();
    mpc.testSampleFromCLR();
    // for (int i = 0; i < 10; i++)
    // {
    //   mpc.planBySamplingSingleConfig("RRTstarkConfigDefault");
    // }
    
    // mpc.planBySamplingSingleConfig("RRTstarkConfigDefault");

  // std::vector<std::string> planners;
  // planners.clear();
  // // planners.push_back("SBLkConfigDefault");
  // // planners.push_back("ESTkConfigDefault");
  // planners.push_back("LBKPIECEkConfigDefault");
  // planners.push_back("BKPIECEkConfigDefault");
  // planners.push_back("KPIECEkConfigDefault");
  // planners.push_back("RRTkConfigDefault");
  // planners.push_back("RRTConnectkConfigDefault");
  // planners.push_back("RRTstarkConfigDefault");
  // // //planners.push_back("TRRTkConfigDefault");   //似乎无法正常运行，move_group崩溃，需要设置额外的参数
  // planners.push_back("PRMkConfigDefault"); 
  // planners.push_back("PRMstarkConfigDefault");
  // mpc.decomposingTest(planners, planners, 100);

  // bool bSuccess = mpc.planBasedPRM("RRTkConfigDefault", 60);
//  bool bSuccess = mpc.planBySamplingSingleConfig("SBLkConfigDefault");
//   if( bSuccess)
//     ROS_INFO("SUCCESS");
//   else
//     ROS_INFO("FAILURE");

/*
  CraneLocationRegions pickCLR(mpc.initPose_);
  CraneLocationRegions placeCLR(mpc.goalPose_);
  Problem* pPro = new Problem("whole", &pickCLR, &placeCLR);
  BiMRRTs planner(pPro);


  bool bSuccess = planner.Plan();
  robot_trajectory::RobotTrajectoryPtr traj;
  traj.reset(new robot_trajectory::RobotTrajectory(mpc.whole_group_->getRobotModel(), "whole"));
  ConstructRobotTrajectory crt(mpc.whole_group_->getRobotModel(), "whole");
  crt.convertPath(planner.Path, *traj);
  if( bSuccess)
    ROS_INFO("SUCCESS");
  else
    ROS_INFO("FAILURE");
  
  moveit_msgs::RobotTrajectory trajectory;
  traj->getRobotTrajectoryMsg(trajectory) ;
  ros::Publisher display_publisher_;
  display_publisher_ = node_handle.advertise<moveit_msgs::DisplayTrajectory>("/move_group/display_planned_path", 1, true);
  moveit_msgs::DisplayTrajectory display_trajectory_;
  moveit_msgs::RobotState rs_msg;
  robot_state::robotStateToRobotStateMsg(traj->getFirstWayPoint(), rs_msg);
  display_trajectory_.trajectory_start = rs_msg;
  display_trajectory_.trajectory.push_back(trajectory);
  while(ros::ok())
  {
    display_publisher_.publish(display_trajectory_);
    sleep(20);
  }
*/
  ros::shutdown();  
  return 0;
}


