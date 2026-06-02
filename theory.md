## 网络结构

简单多层感知机，使用ELU激活函数，每层ELU后dropout

学习一个从z到q的转换，q'=f(z)

## 损失函数

损失函数取

$$
loss=V(net(z))+w*\log{\frac{||f(z)||}{\sigma || z ||}}
$$

第一项是势能极小，第二项对应等比例变换

## 雅各比伪逆

最终运动方程为

$$
\ddot q=-\frac{\partial V}{\partial q}
$$

化简得到

$$
\frac{\partial f}{\partial z_i}\ddot{z_i}=-\left .\frac{\partial V(q)}{\partial q}\right |_{q=f(z)}-\frac{\partial^2 f}{\partial z_i z_j}\dot{z_i}\dot{z_j}
$$

由于方程超静定，只能取

$$
\ddot{z}=\mathbf{argmin}(||\frac{\partial f}{\partial z_i}\ddot{z_i}+\left .\frac{\partial V(q)}{\partial q}\right |_{q=f(z)}+\frac{\partial^2 f}{\partial z_i z_j}\dot{z_i}\dot{z_j}||)
$$

令

$$
J=\frac{\partial f}{\partial z_i}
$$

雅各比伪逆为

$$
J^{\dagger}=(J^TJ+\lambda^2 I)^{-1}J^T
$$

从而

$$
\ddot{z}=J^{\dagger}(-\left .\frac{\partial V(q)}{\partial q}\right |_{q=f(z)}-\frac{\partial^2 f}{\partial z_i\partial z_j}\dot{z_i}\dot{z_j})
$$