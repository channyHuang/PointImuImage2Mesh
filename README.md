# 前言
重建带PointCloud、IMU和Image的ros包生成带颜色的网格模型。

基于[ImMesh](https://github.com/hku-mars/ImMesh.git)进行修改。

依赖于

# 修改笔记
## 源码catkin_make编译成功但运行报错
```sh
ERROR::PROGRAM_LINKING_ERROR of type: PROGRAM
error: vertex shader lacks `main'

 -- --------------------------------------------------- -- 
```
不清楚是否和OpenGL或其它版本有关系，注意到源码使用的ImGui和自己使用的ImGui有差异，把ImGui替换成自己使用的版本，问题解决，能正常运行看到界面显示。

# 后续修改计划
## 