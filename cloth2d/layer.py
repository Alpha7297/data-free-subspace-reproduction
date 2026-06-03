import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
class MyNet(nn.Module):
    def __init__(self,base_output,in_dim,out_dim,hid_dim=64):
        super().__init__()
        self.in_dim=in_dim
        self.out_dim=out_dim
        self.hid_dim=hid_dim
        self.net=nn.Sequential(
            nn.Linear(in_dim,hid_dim),
            nn.ELU(),
            nn.Linear(hid_dim,hid_dim),
            nn.ELU(),
            nn.Linear(hid_dim,hid_dim),
            nn.ELU(),
            nn.Linear(hid_dim,hid_dim),
            nn.ELU(),
            nn.Linear(hid_dim,out_dim)
        )
        self.base_output=base_output
    def forward(self,x,t_schedule=1.0):
        origin_q=self.net(x).reshape(-1,HEIGHT*WIDTH-4,3)# B N 3
        q1=origin_q[:,:WIDTH-2,:]
        q2=origin_q[:,WIDTH-2:WIDTH-2+WIDTH*(HEIGHT-2),:]
        q3=origin_q[:,WIDTH-2+WIDTH*(HEIGHT-2):,:]
        b0=self.base_output[0].reshape(1,1,3).expand(x.shape[0],-1,-1)
        b1=self.base_output[WIDTH-1].reshape(1,1,3).expand(x.shape[0],-1,-1)
        b2=self.base_output[WIDTH*(HEIGHT-1)].reshape(1,1,3).expand(x.shape[0],-1,-1)
        b3=self.base_output[WIDTH*HEIGHT-1].reshape(1,1,3).expand(x.shape[0],-1,-1)
        q1=self.base_output[1:WIDTH-1]+t_schedule*q1
        q2=self.base_output[WIDTH:WIDTH*(HEIGHT-1)]+t_schedule*q2
        q3=self.base_output[WIDTH*(HEIGHT-1)+1:WIDTH*HEIGHT-1]+t_schedule*q3
        return torch.cat((b0,q1,b1,q2,b2,q3,b3),dim=1)# B W*H 3
