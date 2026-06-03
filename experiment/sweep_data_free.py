from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.autograd.functional as F

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from layer import MyNet


LENGTH=100
OUT_DIM=198
K_HOOK=2.0
REST=1.0
DT=0.01
FRAMES=500
DEVICE=torch.device("cpu")
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parent
ITERS=[5000,10000,50000,100000,190000]
INDIMS=[10,20]


def read_series(name):
    data=pd.read_csv(ROOT/name,sep=r"\s+",header=None).to_numpy(dtype=np.float64)
    return data.reshape(data.shape[0],LENGTH,2)


def metric(data_free,explicit):
    n=min(len(data_free),len(explicit))
    diff=data_free[:n]-explicit[:n]
    point_l2=np.linalg.norm(diff,axis=-1)
    frame_rms=np.sqrt(np.mean(point_l2*point_l2,axis=-1))
    return {
        "mean_rms":float(np.mean(frame_rms)),
        "max_rms":float(np.max(frame_rms)),
        "mean_l2":float(np.mean(point_l2)),
        "max_l2":float(np.max(point_l2)),
    }


def base_output():
    return torch.cat((torch.arange(LENGTH,dtype=torch.float32,device=DEVICE)*REST,
                      torch.zeros(LENGTH,device=DEVICE)),dim=-1)


def jacobi_inverse(J):
    m,n=J.shape
    lbd=1e-4
    J_T=J.transpose(-1,-2)
    return torch.inverse(J_T@J+lbd*lbd*torch.eye(n,device=DEVICE))@J_T


def split_xy(q):
    return q[:LENGTH],q[LENGTH:]


def potential(q):
    x,y=split_xy(q)
    dx=x[1:]-x[:-1]
    dy=y[1:]-y[:-1]
    spring_len=torch.sqrt(dx*dx+dy*dy+1e-8)
    return 0.5*K_HOOK*((spring_len-REST)**2).sum(dim=-1)


def solve_z_from_q(net,z,Q):
    z=z.detach().clone().requires_grad_(True)
    optimizer=torch.optim.AdamW([z],lr=1e-2,weight_decay=1e-4)
    for _ in range(10000):
        loss=((net(z[None,:])[0]-Q[:])**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return z.detach()


def solve_vel_z_from_vel_q(net,pos_Z,vel_Q):
    z=pos_Z.detach().clone().requires_grad_(True)
    def f(z_in):
        return net(z_in[None,:])[0]
    J=F.jacobian(f,z)
    return jacobi_inverse(J)@vel_Q


def acceleration(net,pos_Z,vel_Z):
    def f(z):
        return net(z[None,:])[0]
    def h(z):
        return F.jacobian(f,z)@vel_Z
    z=pos_Z.detach().clone().requires_grad_(True)
    q=net(z[None,:])[0]
    f_hook=-F.jacobian(potential,q)
    J=F.jacobian(f,z)
    H=F.jacobian(h,z)@vel_Z
    return (jacobi_inverse(J)@(f_hook-H)).detach()


def run_data_free(in_dim,iter_count):
    base=base_output()
    net=MyNet(in_dim,OUT_DIM,base)
    state=torch.load(REPO/"net"/f"indim-{in_dim}-{iter_count}.pt",map_location=DEVICE)
    net.load_state_dict(state)
    net.eval()

    pos_Q=base.clone()
    vel_Q=torch.zeros(2*LENGTH,device=DEVICE)
    pos_Z=torch.zeros(in_dim,device=DEVICE)
    vel_Z=torch.zeros(in_dim,device=DEVICE)
    for i in range(LENGTH//2):
        vel_Q[i+LENGTH+LENGTH//2]=-1

    pos_Z=solve_z_from_q(net,pos_Z,pos_Q)
    vel_Z=solve_vel_z_from_vel_q(net,pos_Z,vel_Q)
    pos_rows=[]
    vel_rows=[]
    for _ in range(FRAMES):
        pos_rows.append(torch.stack((pos_Q[:LENGTH],pos_Q[LENGTH:]),dim=-1).detach().cpu().numpy())
        vel_rows.append(torch.stack((vel_Q[:LENGTH],vel_Q[LENGTH:]),dim=-1).detach().cpu().numpy())
        def f(z):
            return net(z[None,:])[0]
        acc_Z=acceleration(net,pos_Z,vel_Z)
        vel_Z+=DT*acc_Z
        pos_Z+=DT*vel_Z
        pos_Q=net(pos_Z[None,:])[0]
        vel_Q=F.jacobian(f,pos_Z)@vel_Z
    return np.stack(pos_rows),np.stack(vel_rows)


def table(title,rows,key_prefix):
    out=[f"### {title}","", "| 训练周期 | 平均RMS | 平均L2 | 最大RMS | 最大L2 |", "|---:|---:|---:|---:|---:|"]
    for iter_count in ITERS:
        row=rows.get(iter_count)
        if row is None:
            out.append(f"| {iter_count} | 缺失 | 缺失 | 缺失 | 缺失 |")
        else:
            out.append(f"| {iter_count} | {row[key_prefix+'_mean_rms']:.6g} | {row[key_prefix+'_mean_l2']:.6g} | {row[key_prefix+'_max_rms']:.6g} | {row[key_prefix+'_max_l2']:.6g} |")
    out.append("")
    return "\n".join(out)


def main():
    explicit_pos=read_series("explicit_pos.csv")
    explicit_vel=read_series("explicit_vel.csv")
    all_results={}
    for in_dim in INDIMS:
        all_results[in_dim]={}
        for iter_count in ITERS:
            path=REPO/"net"/f"indim-{in_dim}-{iter_count}.pt"
            if not path.exists():
                all_results[in_dim][iter_count]=None
                print(f"missing {path.name}")
                continue
            print(f"running in_dim={in_dim}, iter={iter_count}")
            pos,vel=run_data_free(in_dim,iter_count)
            pos_m=metric(pos,explicit_pos)
            vel_m=metric(vel,explicit_vel)
            row={
                "pos_mean_rms":pos_m["mean_rms"],
                "pos_mean_l2":pos_m["mean_l2"],
                "pos_max_rms":pos_m["max_rms"],
                "pos_max_l2":pos_m["max_l2"],
                "vel_mean_rms":vel_m["mean_rms"],
                "vel_mean_l2":vel_m["mean_l2"],
                "vel_max_rms":vel_m["max_rms"],
                "vel_max_l2":vel_m["max_l2"],
            }
            all_results[in_dim][iter_count]=row

    print()
    print("data_free 权重扫描结果")
    print("基准为 experiment/explicit_pos.csv 和 experiment/explicit_vel.csv。误差是绝对误差，未乘 100。")
    print()
    for in_dim in INDIMS:
        print(f"## in_dim={in_dim}")
        print()
        print(table("位置误差",all_results[in_dim],"pos"))
        print(table("速度误差",all_results[in_dim],"vel"))


if __name__=="__main__":
    main()
