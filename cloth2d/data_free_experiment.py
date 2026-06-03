from pathlib import Path

import argparse
import re
import numpy as np
import torch

from config import *
from layer import MyNet
from loss import potential
import static_experiment as static

ROOT=Path(__file__).resolve().parent
DEVICE=torch.device("cpu")
DEFAULT_ITERS=[10000,50000,100000,200000,300000]
MAX_ITERS=10000
LR=1.0


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
    for _ in range(10000):
        q=net(z[None,:])[0]
        loss=((q-target_q)**2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return z.detach()


def force_external():
    return torch.tensor(static.EXTERNAL_FORCE,dtype=torch.float32,device=DEVICE).reshape(-1)


def static_energy(net,z,force):
    q=net(z[None,:])[0]
    return potential(q[None,:,:])[0]-(force*q.reshape(-1)).sum()


def solve_data_free_static(model_in_dim,iter_count,max_iters=MAX_ITERS,lr=LR):
    net,base_output,path=load_net(model_in_dim,iter_count)
    z=torch.zeros(model_in_dim,device=DEVICE)
    z=solve_z_from_q(net,z,base_output)
    z=z.detach().clone().requires_grad_(True)
    force=force_external()
    optimizer=torch.optim.LBFGS(
        [z],
        lr=lr,
        max_iter=max_iters,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
    )

    def closure():
        energy=static_energy(net,z,force)
        optimizer.zero_grad()
        energy.backward()
        return energy

    optimizer.step(closure)
    with torch.no_grad():
        q=net(z[None,:])[0].detach().cpu().numpy()
    return q,path


def state_error(data,ref):
    diff=data-ref
    point_l2=np.linalg.norm(diff,axis=-1)
    return {
        "mean_l2":float(np.mean(point_l2)),
        "rms":float(np.sqrt(np.mean(point_l2*point_l2))),
        "max_l2":float(np.max(point_l2)),
    }


def metric_row(model_in_dim,iter_count,pos,ref_pos):
    pos_error=state_error(pos,ref_pos)
    return {
        "in_dim":model_in_dim,
        "iter":iter_count,
        "pos_rms":pos_error["rms"],
        "pos_mean_l2":pos_error["mean_l2"],
        "pos_max_l2":pos_error["max_l2"],
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
    print("| in_dim | iter | 平均RMS | 平均L2 | 最大L2 |")
    print("|---:|---:|---:|---:|---:|")
    for row in rows:
        print(f"| {row['in_dim']} | {row['iter']} | {row['pos_rms']:.6g} | {row['pos_mean_l2']:.6g} | {row['pos_max_l2']:.6g} |")
    print(flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--in-dim",type=int,default=in_dim)
    parser.add_argument("--iters",type=str,default="")
    parser.add_argument("--all",action="store_true")
    parser.add_argument("--max-iters",type=int,default=MAX_ITERS)
    parser.add_argument("--lr",type=float,default=LR)
    parser.add_argument("--static-max-iters",type=int,default=static.MAX_ITERS)
    parser.add_argument("--grad-threshold",type=float,default=static.GRAD_THRESHOLD)
    args=parser.parse_args()
    if args.all:
        iters=existing_iters(args.in_dim)
    else:
        iters=parse_iters(args.iters)
    ref_pos,ref_result=static.solve(args.static_max_iters,args.grad_threshold)
    print()
    print("data_free static equilibrium error")
    print("reference=static_experiment.solve")
    print(f"reference_success={ref_result.success}")
    print(f"reference_iters={ref_result.nit}")
    print(f"data_free_max_iters={args.max_iters}")
    for iter_count in iters:
        path=ROOT/"net"/f"indim-{args.in_dim}-{iter_count}.pt"
        if not path.exists():
            print(f"missing {path}")
            continue
        print(f"running in_dim={args.in_dim},iter={iter_count}")
        pos,_=solve_data_free_static(args.in_dim,iter_count,args.max_iters,args.lr)
        row=metric_row(args.in_dim,iter_count,pos,ref_pos)
        print_table([row])


if __name__=="__main__":
    main()
