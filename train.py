import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from layer import MyNet
from loss import loss
print(torch.cuda.is_available())
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
LENGTH=100
in_dim=20
out_dim=198
k_hook=2.0
leng_origin=1.0
base_output=torch.cat((torch.arange(LENGTH,dtype=torch.float32,device=device)*leng_origin,
                       torch.zeros(LENGTH,device=device)),dim=-1)
Path("net").mkdir(exist_ok=True)

def split_xy(q,length):
    return q[:,:length],q[:,length:]

def potential(q):
    x,y=split_xy(q,LENGTH)
    dx=x[:,1:]-x[:,:-1]
    dy=y[:,1:]-y[:,:-1]
    spring_len=torch.sqrt(dx*dx+dy*dy+1e-8)
    spring_energy=0.5*k_hook*((spring_len-leng_origin)**2).sum(dim=-1)
    return spring_energy

net=MyNet(in_dim,out_dim,base_output).to(device)

optimizer=torch.optim.AdamW(
    net.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

n_train_iters=5000
batch_size=32

for i in range(n_train_iters):
    t_schedule=i/n_train_iters
    z=torch.randn(batch_size,in_dim,device=device)
    loss_value=loss(
        net=net,
        z=z,
        t_schedule=t_schedule,
        potential=potential,
        sigma=1.0,
        weight_expand=0.25,
    )
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()

    if i%1000==0:
        print(i,loss_value.item())
    if i%10000==0:
        torch.save(net.state_dict(),f"net/indim-10-{i}.pt")
torch.save(net.state_dict(),f"net/indim-20-{n_train_iters}.pt")
