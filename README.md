# 前言
重建带PointCloud、IMU和Image的ros包生成带颜色的网格模型。

基于[ImMesh](https://github.com/hku-mars/ImMesh.git)进行修改。

依赖于[livox_ros_driver Installation](https://github.com/Livox-SDK/livox_ros_driver)

# 修改笔记
## 源码catkin_make编译成功但运行报错
```sh
ERROR::PROGRAM_LINKING_ERROR of type: PROGRAM
error: vertex shader lacks `main'

 -- --------------------------------------------------- -- 
```
不清楚是否和OpenGL或其它版本有关系，注意到源码使用的ImGui和自己使用的ImGui有差异，把ImGui替换成自己使用的版本，问题解决，能正常运行看到界面显示。

## 参考R3Live增加从Image中获取点云颜色
需要知道对应输入数据中摄像机相对于激光雷达的外参，即摄像机到激光雷达的旋转矩阵和平移向量，或激光雷达到摄像机的旋转矩阵和平移向量。

以Kitti数据为测试样例 [KITTI-dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php):
```sh
roslaunch ImMesh mapping_ntlc.launch
rosbag play kitti_07.bag
``` 

# 后续修改计划
## 直接按字节和格式解析ros包以去除ros编译环境限制
## 增加网格纹理映射替代当前的点云颜色
