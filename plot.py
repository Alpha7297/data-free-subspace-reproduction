import numpy as np
from layer import MyNet
import torch
import torch.nn as nn
import torch.autograd.functional as F
device=torch.device("cpu")
LENGTH=100
in_dim=10
out_dim=200
k_hook=2.0
leng_origin=1.0
base_output=torch.cat((torch.arange(LENGTH,dtype=torch.float32,device=device)*leng_origin,
                       torch.zeros(LENGTH,device=device)),dim=-1)
net=MyNet(in_dim,out_dim,base_output)
state_dict=torch.load("net/mlp.pt",map_location=device)
net.load_state_dict(state_dict)
net.eval()
pos_Q=base_output.clone()
vel_Q=torch.zeros(2*LENGTH,device=device)
pos_Z=torch.zeros(in_dim,device=device)
vel_Z=torch.zeros(in_dim,device=device)
for i in range(LENGTH//2):
    vel_Q[i+LENGTH+LENGTH//2]=-1
# 反解pos_Z
def solve_z_from_q(net,z,Q):
    z=z.detach().clone().requires_grad_(True)
    lr=1e-3
    for _ in range(10000):
        q=net(z[None,:])
        loss=((q-Q[None,:])**2).mean()
        if loss.item()<1e-5:
            break
        if z.grad is not None:
            z.grad.zero_()
        loss.backward()
        with torch.no_grad():
            z-=z.grad*lr
    return z.detach()

# Jacobi inverse
def Jacobi_inverse(J):
    m,n=J.shape
    lbd=1e-4
    J_T=J.transpose(-1,-2)
    J_dagger=torch.inverse(J_T@J+lbd*lbd*torch.eye(n,device=device))@J_T
    return J_dagger

# 反解vel_Z
def solve_vel_z_from_vel_q(net,pos_Z,vel_Q):
    z=pos_Z.detach().clone().requires_grad_(True)
    def f(z_in):
        return net(z_in[None,:])[0]
    J=F.jacobian(f,z)
    return Jacobi_inverse(J)@vel_Q

def split_xy(q,length):
    return q[:length],q[length:]

def potential(q):
    x,y=split_xy(q,LENGTH)
    dx=x[1:]-x[:-1]
    dy=y[1:]-y[:-1]
    spring_len=torch.sqrt(dx*dx+dy*dy+1e-8)
    spring_energy=0.5*k_hook*((spring_len-leng_origin)**2).sum(dim=-1)
    return spring_energy

pos_Z=solve_z_from_q(net,pos_Z,pos_Q)
vel_Z=solve_vel_z_from_vel_q(net,pos_Z,vel_Q)

def acceleration(net,pos_Z:torch.tensor,vel_Z:torch.tensor):
    def f(z):
        return net(z[None,:])[0]
    def h(z):
        return F.jacobian(f,z)@vel_Z
    z=pos_Z.detach().clone().requires_grad_(True)
    q=net(z[None,:])[0]
    f_hook=-F.jacobian(potential,q)
    J=F.jacobian(f,z)
    H=F.jacobian(h,z)@vel_Z
    J_dagger=Jacobi_inverse(J)
    acc_Z=J_dagger@(f_hook-H)
    return acc_Z.detach()

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

dt=0.3
n_frames=500

def q_to_xy(q):
    q=q.detach().cpu()
    return q[:LENGTH],q[LENGTH:]

fig,ax=plt.subplots()
q=net(pos_Z[None,:])[0]
x,y=q_to_xy(q)
line,=ax.plot(x,y,"o-",markersize=3,linewidth=1)
pin=ax.scatter([x[0]],[y[0]],c="red",s=40,zorder=3)
ax.set_aspect("equal",adjustable="box")
ax.set_xlim(-2,LENGTH*leng_origin+2)
ax.set_ylim(-20,20)
ax.grid(True,alpha=0.3)

def update(frame):
    global pos_Z,vel_Z
    acc_Z=acceleration(net,pos_Z,vel_Z)
    vel_Z=vel_Z+dt*acc_Z
    pos_Z=pos_Z+dt*vel_Z
    q=net(pos_Z[None,:])[0]
    x,y=q_to_xy(q)
    line.set_data(x,y)
    pin.set_offsets([[x[0],y[0]]])
    ax.set_title(f"frame={frame}, dt={dt:g}")
    return line,pin

anim=FuncAnimation(fig,update,frames=n_frames,interval=dt*10,blit=False)
fig._anim=anim
plt.show()
