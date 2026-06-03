# 文件夹说明

fix_t_loss_net

```python
t_schedule=1
E_norm=((torch.log(t_schedule*dist_x*sigma+eps)-torch.log(dist_q+eps))).mean(dim=-1)
```

fix_t_net

```python
t_schedule=1
E_norm=((torch.log(t_schedule*dist_x*sigma+eps)-torch.log(dist_q+eps))**2).mean(dim=-1)
```

t_loss_net

```python
t_schedule=min(1.0,2.0*epoch/num_epoches)
E_norm=((torch.log(t_schedule*dist_x*sigma+eps)-torch.log(dist_q+eps))).mean(dim=-1)
```

t_net

```python
t_schedule=min(1.0,2.0*epoch/num_epoches)
E_norm=((torch.log(t_schedule*dist_x*sigma+eps)-torch.log(dist_q+eps))**2).mean(dim=-1)
```

实验结果是，t_loss_net和fix_t_loss_net显著优于另外两个

并且fix_t_loss_net优于t_loss_net，前者具有优秀的中心对称性，而后者没有

而原论文中的正是t_net，实际在这个网络下实验中原论文中的仿真结果完全非物理，说明这个网络在这个弹簧系统下有问题