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
        if base_output is not None and base_output.numel()==out_dim+2:
            full_dim=base_output.numel()
            fixed=torch.tensor([0,full_dim//2],device=base_output.device)
            mask=torch.ones(full_dim,dtype=torch.bool,device=base_output.device)
            mask[fixed]=False
            self.register_buffer("free_inds",torch.nonzero(mask,as_tuple=False).flatten())
        else:
            self.free_inds=None
    def forward(self,z,t_schedule=1.0):
        if self.base_output is None:
            return self.net(z)
        elif self.free_inds is not None:
            dq=self.net(z)
            q=self.base_output.expand(z.shape[0],-1).clone()
            q[:,self.free_inds]=self.base_output[self.free_inds]+t_schedule*dq
            return q
        else:
            return self.base_output+t_schedule*self.net(z)
