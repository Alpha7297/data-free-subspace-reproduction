# Data-Free Learning of Reduced-Order Kinematics论文复现

## requirements

pytorch numpy scipy matplotlib

## 项目结构

cloth2d为2D弹簧质点网格，复杂系统用于测试静力学特征

cloth2d/plot.py 使用matplotlib展现仿真结果

line1d为一维弹簧链，简单系统用于测试动力学特征

## 项目说明

根据我的研究，论文中的方法实际上是学习了稳定态附近的位形空间在低维的投影，因此直接用动力学过程跟传统方法比较并不合理，误差也很大

实际上，base_output的作用也不是直接给出受力平衡态，而是给一个不会崩塌的初始条件，平衡态是未知的，是"data_free"的，也是网络需要学习的

正确的实验比较方法是，在训练得到低维空间后，施加一定大小外力，在低维空间下求解受力平衡态，并投影回实际空间中，分析实际空间中是否达到了受力平衡，以及和传统方法之间误差多少