import torch
from layer import MyNet
import torch.nn as nn
import torch.nn.functional as F

def loss(net:MyNet,z,t_schedule,potential,sigma=1.0,weight_expand=0.25,eps=1e-8,project_q=None):
    eps=1e-8
    q=net.forward(z,t_schedule)
    if project_q is not None:
        q=project_q(q)
    E_pot=potential(q)
    z_delta=z[:,None,:]-z[None,:,:]# B B in_dim
    z_dist=(z_delta**2).sum(dim=-1)# B B
    q_delta=q[:,None,:]-q[None,:,:]# B B out_dim
    q_dist=0.5*(q_delta**2).sum(dim=-1)+eps# B B
    factor=torch.log(t_schedule*sigma*z_dist+eps)-torch.log(q_dist)# B B
    E_expand=(factor**2).sum(dim=-1)# B
    return E_pot.mean()+E_expand.mean()*weight_expand
