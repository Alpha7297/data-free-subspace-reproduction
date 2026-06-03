import torch
import torch.nn as nn
from layer import MyNet
from config import *
from loss import loss
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def init_pos():
    i=torch.arange(HEIGHT,dtype=torch.float32,device=device)
    j=torch.arange(WIDTH,dtype=torch.float32,device=device)
    I,J=torch.meshgrid(i,j,indexing="ij")
    K=torch.zeros_like(I)
    return torch.stack((I.T,J.T,K),dim=-1).reshape(-1,3)
base_output=init_pos()
net=MyNet(base_output,in_dim,out_dim,hid_dim).to(device)

optimizer=torch.optim.AdamW(
    net.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

num_epoches=100000
batch_size=32

for epoch in range(num_epoches):
    t_schedule=epoch/num_epoches
    z=torch.randn(batch_size,in_dim,device=device,dtype=torch.float32)
    loss_value=loss(
        net=net,z=z,t_schedule=t_schedule
    )
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    if epoch%1000==0:
        print(f"{epoch} {loss_value.item()}")
    if epoch%10000==0:
        torch.save(net.state_dict(),f"cloth2d/net/indim-20-{epoch}.pt")