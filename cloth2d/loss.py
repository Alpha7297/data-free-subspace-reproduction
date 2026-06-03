import torch
import torch.nn as nn
from layer import MyNet
from config import *
norm2weight=0.25
sigma=1.0
eps=1e-8

def potential(q):
    # [B,HEIGHT*WIDTH,3]
    reshaped_q=q.reshape(-1,HEIGHT,WIDTH,3)# [B,H,W,3]
    dx=reshaped_q[:,1:,:,:]-reshaped_q[:,:-1,:,:]# [B,H-1,W,3]
    dy=reshaped_q[:,:,1:,:]-reshaped_q[:,:,:-1,:]# [B,H,W-1,3]
    len_x=torch.sqrt((dx*dx).sum(dim=-1)+1e-8)# [B,H-1,W]
    len_y=torch.sqrt((dy*dy).sum(dim=-1)+1e-8)# [B,H,W-1]
    energy_x=((len_x-origin_len)**2).sum(dim=(1,2))# [B]
    energy_y=((len_y-origin_len)**2).sum(dim=(1,2))# [B]
    return 0.5*k_hook*(energy_x+energy_y)# [B]

def loss(net:nn.Module,z,t_schedule):
    q=net(z,t_schedule=t_schedule)
    E_spring=potential(q)
    new_q=q.reshape(-1,HEIGHT*WIDTH,3)
    diff_q=new_q[None,:,:,:]-new_q[:,None,:,:]# [B,B,WH,3]
    diff_x=z[None,:,:]-z[:,None,:]# [B,B,in_dim]
    dist_q=(diff_q**2).sum(dim=(2,3))
    dist_x=(diff_x**2).sum(dim=-1)
    E_norm=((torch.log(t_schedule*dist_x*sigma+eps)-torch.log(dist_q+eps))**2).mean(dim=-1)
    return (E_spring+norm2weight*E_norm).mean(dim=-1)