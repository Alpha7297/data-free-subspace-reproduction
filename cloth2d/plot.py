from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.autograd.functional as AF

from config import *
from layer import MyNet
from loss import potential


device=torch.device("cpu")
ROOT=Path(__file__).resolve().parents[1]
MODEL_PATH=ROOT/"cloth2d"/"net"/f"indim-{in_dim}-200000.pt"
DT=0.1
FRAMES=500
Z_FORCE=3.0
DAMPING=0.995
FORCE_IDS=[(HEIGHT//2)*WIDTH+(WIDTH//2)]
FREE_IDS=[i for i in range(HEIGHT*WIDTH) if i not in fixed_id]
FREE_ID_MAP={i:n for n,i in enumerate(FREE_IDS)}


def init_pos():
    i=torch.arange(HEIGHT,dtype=torch.float32,device=device)
    j=torch.arange(WIDTH,dtype=torch.float32,device=device)
    I,J=torch.meshgrid(i,j,indexing="ij")
    K=torch.zeros_like(I)
    return torch.stack((I.T,J.T,K),dim=-1).reshape(-1,3)


def init_pos_np():
    i=np.arange(HEIGHT,dtype=np.float64)
    j=np.arange(WIDTH,dtype=np.float64)
    I,J=np.meshgrid(i,j,indexing="ij")
    K=np.zeros_like(I)
    return np.stack((I.T,J.T,K),axis=-1).reshape(-1,3)


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
    for _ in range(50000):
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


def data_free_acceleration(net,pos_Z,vel_Z):
    z=pos_Z.detach().clone().requires_grad_(True)
    def f(z_in):
        return net(z_in[None,:])[0]
    def f_flat(z_in):
        return f(z_in).reshape(-1)
    q=f(z)
    force_internal=-AF.jacobian(lambda q_in:potential(q_in[None,:,:])[0],q).reshape(-1)
    force_q=force_internal+force_external()
    J=AF.jacobian(f_flat,z)
    return (jacobi_inverse(J)@force_q).detach()


def spring_pairs():
    pairs=[]
    for i in range(HEIGHT):
        for j in range(WIDTH):
            idx=i*WIDTH+j
            if i+1<HEIGHT:
                pairs.append((idx,(i+1)*WIDTH+j))
            if j+1<WIDTH:
                pairs.append((idx,i*WIDTH+j+1))
    return pairs


SPRING_PAIRS=spring_pairs()


def free_id(i,d):
    return FREE_ID_MAP[i]*3+d


def implicit_external_force_np():
    force=np.zeros((HEIGHT*WIDTH,3),dtype=np.float64)
    if FORCE_IDS is None:
        force[:,2]=Z_FORCE
        force[fixed_id,2]=0.0
    else:
        force[FORCE_IDS,2]=Z_FORCE
    return force


def internal_force_and_blocks(pos):
    force=np.zeros_like(pos)
    blocks=[]
    eye=np.eye(3,dtype=np.float64)
    for a,b in SPRING_PAIRS:
        l=pos[a]-pos[b]
        norm=np.sqrt(np.dot(l,l))+1e-8
        ratio=(norm-origin_len)/norm
        spring_force=k_hook*ratio*l
        force[a]-=spring_force
        force[b]+=spring_force
        outer=np.outer(l,l)
        D=-k_hook*(ratio*eye+origin_len*outer/(norm**3))
        blocks.extend(((a,a,D),(a,b,-D),(b,a,-D),(b,b,D)))
    return force,blocks


def implicit_step(pos,vel):
    force,blocks=internal_force_and_blocks(pos)
    force+=implicit_external_force_np()
    rows=[]
    cols=[]
    data=[]
    bvec=np.zeros(len(FREE_IDS)*3,dtype=np.float64)
    for bi,bj,block in blocks:
        if bi in fixed_id or bj in fixed_id:
            continue
        for r in range(3):
            for c in range(3):
                rows.append(free_id(bi,r))
                cols.append(free_id(bj,c))
                data.append(-DT*DT*block[r,c])
    vel=DAMPING*vel
    for i in FREE_IDS:
        for d in range(3):
            idx=free_id(i,d)
            rows.append(idx)
            cols.append(idx)
            data.append(1.0)
            bvec[idx]=DT*vel[i,d]+DT*DT*force[i,d]
    size=len(FREE_IDS)*3
    A=sp.coo_matrix((data,(rows,cols)),shape=(size,size)).tocsr()
    dq=spla.spsolve(A,bvec).reshape(len(FREE_IDS),3)
    pos=pos.copy()
    vel=vel.copy()
    pos[FREE_IDS]+=dq
    vel[FREE_IDS]=dq/DT
    pos[fixed_id]=init_pos_np()[fixed_id]
    vel[fixed_id]=0.0
    return pos,vel


def draw_state(ax,q,title):
    if isinstance(q,torch.Tensor):
        q=q.detach().cpu().numpy()
    center=(HEIGHT//2)*WIDTH+(WIDTH//2)
    X=q[:,0].reshape(WIDTH,HEIGHT).T
    Y=q[:,1].reshape(WIDTH,HEIGHT).T
    Z=q[:,2].reshape(WIDTH,HEIGHT).T
    ax.clear()
    ax.plot_surface(X,Y,Z,cmap="viridis",linewidth=0,antialiased=True)
    ax.scatter(q[fixed_id,0],q[fixed_id,1],q[fixed_id,2],c="red",s=30)
    ax.set_xlim(0,HEIGHT)
    ax.set_ylim(0,WIDTH)
    ax.set_zlim(-10,10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{title},center_z={q[center,2]:.4g}")


net,base_output=load_net()
data_free_pos_Z=torch.zeros(in_dim,device=device)
data_free_vel_Z=torch.zeros(in_dim,device=device)
data_free_pos_Z=solve_z_from_q(net,data_free_pos_Z,base_output)
implicit_pos=init_pos_np()
implicit_vel=np.zeros_like(implicit_pos)

fig=plt.figure(figsize=(12,6))
ax_implicit=fig.add_subplot(121,projection="3d")
ax_data_free=fig.add_subplot(122,projection="3d")
draw_state(ax_implicit,implicit_pos,"implicit")
draw_state(ax_data_free,net(data_free_pos_Z[None,:])[0],"data_free")
ax_implicit.view_init(elev=25,azim=-60)
ax_data_free.view_init(elev=25,azim=-60)
fig.tight_layout()


def update(frame):
    global implicit_pos,implicit_vel,data_free_pos_Z,data_free_vel_Z
    implicit_pos,implicit_vel=implicit_step(implicit_pos,implicit_vel)
    acc_Z=data_free_acceleration(net,data_free_pos_Z,data_free_vel_Z)
    data_free_vel_Z=DAMPING*data_free_vel_Z+DT*acc_Z
    data_free_pos_Z=data_free_pos_Z+DT*data_free_vel_Z
    q_data_free=net(data_free_pos_Z[None,:])[0]
    draw_state(ax_implicit,implicit_pos,f"implicit frame={frame}")
    draw_state(ax_data_free,q_data_free,f"data_free frame={frame}")


anim=FuncAnimation(fig,update,frames=FRAMES,interval=DT*100,blit=False)
fig._anim=anim
plt.show()
