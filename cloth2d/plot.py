from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch
import torch.autograd.functional as AF

from config import *
from layer import MyNet
from loss import potential


device=torch.device("cpu")
ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/"cloth2d"/"net"/f"indim-{in_dim}-90000.pt"
DT=0.01
FRAMES=500
Z_FORCE=5.0
DAMPING=0.995
FORCE_IDS=[(HEIGHT//2)*WIDTH+(WIDTH//2)]


def init_pos():
    i=torch.arange(HEIGHT,dtype=torch.float32,device=device)
    j=torch.arange(WIDTH,dtype=torch.float32,device=device)
    I,J=torch.meshgrid(i,j,indexing="ij")
    K=torch.zeros_like(I)
    return torch.stack((I.T,J.T,K),dim=-1).reshape(-1,3)


def load_net():
    base_output=init_pos()
    net=MyNet(base_output,in_dim,out_dim,hid_dim).to(device)
    state_dict=torch.load(MODEL_PATH,map_location=device)
    net.load_state_dict(state_dict)
    net.eval()
    return net,base_output


def solve_z_from_q(net,z,target_q):
    z=z.detach().clone().requires_grad_(True)
    optimizer=torch.optim.AdamW([z],lr=1e-2,weight_decay=1e-4)
    for _ in range(5000):
        q=net(z[None,:])[0]
        loss=((q-target_q)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return z.detach()


def jacobi_inverse(J):
    m,n=J.shape
    lbd=1e-2
    J_T=J.transpose(-1,-2)
    return torch.inverse(J_T@J+lbd*lbd*torch.eye(n,device=device))@J_T


def force_external():
    force=torch.zeros(HEIGHT*WIDTH,3,device=device)
    if FORCE_IDS is None:
        force[:,2]=Z_FORCE
        force[fixed_id,2]=0.0
    else:
        force[FORCE_IDS,2]=Z_FORCE
    return force.reshape(-1)


def acceleration(net,pos_Z,vel_Z):
    z=pos_Z.detach().clone().requires_grad_(True)
    def f(z_in):
        return net(z_in[None,:])[0]
    def f_flat(z_in):
        return f(z_in).reshape(-1)
    q=f(z)
    force_internal=-AF.jacobian(lambda q_in:potential(q_in[None,:,:])[0],q).reshape(-1)
    force_q=force_internal+force_external()
    J=AF.jacobian(f_flat,z)
    acc_Z=jacobi_inverse(J)@force_q
    return acc_Z.detach()


def draw_state(ax,q):
    q=q.detach().cpu()
    X=q[:,0].reshape(WIDTH,HEIGHT).T
    Y=q[:,1].reshape(WIDTH,HEIGHT).T
    Z=q[:,2].reshape(WIDTH,HEIGHT).T
    ax.clear()
    ax.plot_surface(X.numpy(),Y.numpy(),Z.numpy(),cmap="viridis",linewidth=0,antialiased=True)
    ax.scatter(q[fixed_id,0],q[fixed_id,1],q[fixed_id,2],c="red",s=30)
    ax.set_xlim(0,HEIGHT)
    ax.set_ylim(0,WIDTH)
    ax.set_zlim(-10,10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


net,base_output=load_net()
pos_Z=torch.zeros(in_dim,device=device)
vel_Z=torch.zeros(in_dim,device=device)
pos_Z=solve_z_from_q(net,pos_Z,base_output)

fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
draw_state(ax,net(pos_Z[None,:])[0])


def update(frame):
    global pos_Z,vel_Z
    acc_Z=acceleration(net,pos_Z,vel_Z)
    vel_Z=DAMPING*vel_Z+DT*acc_Z
    pos_Z=pos_Z+DT*vel_Z
    q=net(pos_Z[None,:])[0]
    draw_state(ax,q)
    ax.set_title(f"frame={frame},dt={DT:g},force_z={Z_FORCE:g}")


anim=FuncAnimation(fig,update,frames=FRAMES,interval=DT*10,blit=False)
fig._anim=anim
plt.show()
