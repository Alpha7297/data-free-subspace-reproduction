import torch
import torch.nn as nn
import torch.nn.functional as F

class MyNet(nn.Module):
    def __init__(self,in_dim,out_dim,base_output):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim,64),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(64,64),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(64,64),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(64,64),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(64,out_dim)
        )
        self.register_buffer("base_output",base_output)
    def forward(self,z,t_schedule=1.0):
        if self.base_output is None:
            return self.net(z)
        else:
            return self.base_output+t_schedule*self.net(z)