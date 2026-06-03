from pathlib import Path

import argparse
import re
import numpy as np
import torch
import torch.autograd.functional as AF

from config import *
from layer import MyNet
from loss import potential
import implicit_experiment as implicit


ROOT=Path(__file__).resolve().parent
DEVICE=torch.device("cpu")
DT=implicit.DT
FRAMES=implicit.FRAMES
DAMPING=implicit.DAMPING
DEFAULT_ITERS=[10000,50000,100000,200000,300000]


def init_pos():
    i=torch.arange(HEIGHT,dtype=torch.float32,device=DEVICE)
    j=torch.arange(WIDTH,dtype=torch.float32,device=DEVICE)
    I,J=torch.meshgrid(i,j,indexing="ij")
    K=torch.zeros_like(I)
    return torch.stack((I.T,J.T,K),dim=-1).reshape(-1,3)


def load_net(model_in_dim,iter_count):
    base_output=init_pos()
    net=MyNet(base_output,model_in_dim,out_dim,hid_dim).to(DEVICE)
    path=ROOT/"net"/f"indim-{model_in_dim}-{iter_count}.pt"
    state_dict=torch.load(path,map_location=DEVICE)
    net.load_state_dict(state_dict)
    net.eval()
    return net,base_output,path


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
    lbd=1e-1
    J_T=J.transpose(-1,-2)
    return torch.inverse(J_T@J+lbd*lbd*torch.eye(n,device=DEVICE))@J_T


def force_external():
    force=torch.zeros(HEIGHT*WIDTH,3,device=DEVICE)
    if implicit.FORCE_IDS is None:
        force[:,2]=implicit.Z_FORCE
        force[fixed_id,2]=0.0
    else:
        force[implicit.FORCE_IDS,2]=implicit.Z_FORCE
    return force.reshape(-1)


def acceleration(net,pos_Z):
    z=pos_Z.detach().clone().requires_grad_(True)
    def f_flat(z_in):
        return net(z_in[None,:])[0].reshape(-1)
    q=net(z[None,:])[0]
    force_internal=-AF.jacobian(lambda q_in:potential(q_in[None,:,:])[0],q).reshape(-1)
    force_q=force_internal+force_external()
    J=AF.jacobian(f_flat,z)
    return (jacobi_inverse(J)@force_q).detach()


def run_data_free(model_in_dim,iter_count,frames):
    net,base_output,path=load_net(model_in_dim,iter_count)
    pos_Z=torch.zeros(model_in_dim,device=DEVICE)
    vel_Z=torch.zeros(model_in_dim,device=DEVICE)
    pos_Z=solve_z_from_q(net,pos_Z,base_output)
    pos_rows=[]
    vel_rows=[]
    status="ok"
    for _ in range(frames):
        acc_Z=acceleration(net,pos_Z)
        if not torch.isfinite(acc_Z).all():
            status="nonfinite_acceleration"
            break
        vel_Z=DAMPING*vel_Z+DT*acc_Z
        pos_Z=pos_Z+DT*vel_Z
        q=net(pos_Z[None,:])[0]
        if not torch.isfinite(q).all() or not torch.isfinite(vel_Z).all():
            status="nonfinite_state"
            break
        def f_flat(z_in):
            return net(z_in[None,:])[0].reshape(-1)
        J=AF.jacobian(f_flat,pos_Z)
        vel_Q=(J@vel_Z).reshape(HEIGHT*WIDTH,3)
        pos_rows.append(q.detach().cpu().numpy())
        vel_rows.append(vel_Q.detach().cpu().numpy())
    if not pos_rows:
        nan_pos=np.full((1,HEIGHT*WIDTH,3),np.nan,dtype=np.float64)
        nan_vel=np.full((1,HEIGHT*WIDTH,3),np.nan,dtype=np.float64)
        return nan_pos,nan_vel,path,status
    return np.stack(pos_rows),np.stack(vel_rows),path,status


def state_error(data,ref):
    diff=data-ref
    point_l2=np.linalg.norm(diff,axis=-1)
    return {
        "mean_l2":float(np.mean(point_l2)),
        "rms":float(np.sqrt(np.mean(point_l2*point_l2))),
        "max_l2":float(np.max(point_l2)),
        "mean_z":float(np.mean(np.abs(diff[:,2]))),
        "max_z":float(np.max(np.abs(diff[:,2]))),
        "center_z_error":float(abs(data[(HEIGHT//2)*WIDTH+(WIDTH//2),2]-ref[(HEIGHT//2)*WIDTH+(WIDTH//2),2])),
    }


def metric_row(model_in_dim,iter_count,pos,vel,ref_pos,ref_vel):
    pos_error=state_error(pos[-1],ref_pos[-1])
    vel_error=state_error(vel[-1],ref_vel[-1])
    eq=implicit.equilibrium_metric(pos[-1],vel[-1])
    ref_eq=implicit.equilibrium_metric(ref_pos[-1],ref_vel[-1])
    return {
        "in_dim":model_in_dim,
        "iter":iter_count,
        "pos_rms":pos_error["rms"],
        "pos_mean_l2":pos_error["mean_l2"],
        "pos_max_l2":pos_error["max_l2"],
        "z_mean_abs":pos_error["mean_z"],
        "z_max_abs":pos_error["max_z"],
        "center_z_error":pos_error["center_z_error"],
        "vel_rms":vel_error["rms"],
        "vel_mean_l2":vel_error["mean_l2"],
        "force_rms":eq["rms_force"],
        "force_rms_ref":ref_eq["rms_force"],
        "force_rms_delta":eq["rms_force"]-ref_eq["rms_force"],
        "center_z":eq["center_z"],
        "center_z_ref":ref_eq["center_z"],
        "status":"ok" if len(pos)==len(ref_pos) and np.isfinite(pos[-1]).all() else "failed",
    }


def parse_iters(raw):
    if raw:
        return [int(x) for x in raw.split(",") if x]
    return DEFAULT_ITERS


def existing_iters(model_in_dim):
    result=[]
    pattern=re.compile(rf"indim-{model_in_dim}-(\d+)\.pt$")
    for path in (ROOT/"net").glob(f"indim-{model_in_dim}-*.pt"):
        match=pattern.match(path.name)
        if match:
            result.append(int(match.group(1)))
    return sorted(result)


def print_table(rows):
    print("| in_dim | iter | status | pos_rms | pos_mean_l2 | pos_max_l2 | z_mean_abs | z_max_abs | center_z_error | vel_rms | force_rms | force_rms_delta | center_z |")
    print("|---:|---:|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        print(f"| {row['in_dim']} | {row['iter']} | {row['status']} | {row['pos_rms']:.6g} | {row['pos_mean_l2']:.6g} | {row['pos_max_l2']:.6g} | {row['z_mean_abs']:.6g} | {row['z_max_abs']:.6g} | {row['center_z_error']:.6g} | {row['vel_rms']:.6g} | {row['force_rms']:.6g} | {row['force_rms_delta']:.6g} | {row['center_z']:.6g} |")
    print(flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--frames",type=int,default=FRAMES)
    parser.add_argument("--in-dim",type=int,default=in_dim)
    parser.add_argument("--iters",type=str,default="")
    parser.add_argument("--all",action="store_true")
    args=parser.parse_args()
    if args.all:
        iters=existing_iters(args.in_dim)
    else:
        iters=parse_iters(args.iters)
    ref_pos,ref_vel=implicit.run(args.frames)
    print()
    print(f"data_free {args.frames}-frame final-state error")
    print("reference=implicit_experiment.run")
    print(f"frames={args.frames}")
    for iter_count in iters:
        path=ROOT/"net"/f"indim-{args.in_dim}-{iter_count}.pt"
        if not path.exists():
            print(f"missing {path}")
            continue
        print(f"running in_dim={args.in_dim},iter={iter_count}")
        pos,vel,_,status=run_data_free(args.in_dim,iter_count,args.frames)
        row=metric_row(args.in_dim,iter_count,pos,vel,ref_pos,ref_vel)
        if status!="ok":
            row["status"]=status
        print_table([row])


if __name__=="__main__":
    main()
