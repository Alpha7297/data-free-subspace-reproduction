from pathlib import Path

import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from config import *


ROOT=Path(__file__).resolve().parent
DT=0.01
FRAMES=100
Z_FORCE=5.0
DAMPING=0.995
FORCE_IDS=[(HEIGHT//2)*WIDTH+(WIDTH//2)]
FREE_IDS=[i for i in range(HEIGHT*WIDTH) if i not in fixed_id]
FREE_ID_MAP={i:n for n,i in enumerate(FREE_IDS)}


def init_pos():
    i=np.arange(HEIGHT,dtype=np.float64)
    j=np.arange(WIDTH,dtype=np.float64)
    I,J=np.meshgrid(i,j,indexing="ij")
    K=np.zeros_like(I)
    return np.stack((I.T,J.T,K),axis=-1).reshape(-1,3)


def free_id(i,d):
    return FREE_ID_MAP[i]*3+d


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


def external_force():
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


def write_series(path,series):
    with open(path,"w",encoding="utf-8") as f:
        for q in series:
            for p in q:
                f.write(f"{p[0]} {p[1]} {p[2]} ")
            f.write("\n")


def equilibrium_metric(pos,vel):
    force,_=internal_force_and_blocks(pos)
    force+=external_force()
    free_force=force[FREE_IDS]
    free_vel=vel[FREE_IDS]
    force_l2=np.linalg.norm(free_force,axis=-1)
    vel_l2=np.linalg.norm(free_vel,axis=-1)
    return {
        "mean_force":float(np.mean(force_l2)),
        "max_force":float(np.max(force_l2)),
        "rms_force":float(np.sqrt(np.mean(force_l2*force_l2))),
        "mean_vel":float(np.mean(vel_l2)),
        "max_vel":float(np.max(vel_l2)),
        "center_z":float(pos[(HEIGHT//2)*WIDTH+(WIDTH//2),2]),
    }


def implicit_step(pos,vel):
    force,blocks=internal_force_and_blocks(pos)
    force+=external_force()
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
    pos[fixed_id]=init_pos()[fixed_id]
    vel[fixed_id]=0.0
    return pos,vel


def run(frames=FRAMES):
    pos=init_pos()
    vel=np.zeros_like(pos)
    pos_rows=[]
    vel_rows=[]
    for _ in range(frames):
        pos,vel=implicit_step(pos,vel)
        pos_rows.append(pos.copy())
        vel_rows.append(vel.copy())
    return np.stack(pos_rows),np.stack(vel_rows)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--frames",type=int,default=FRAMES)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"experiment")
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pos,vel=run(args.frames)
    pos_path=args.output_dir/f"implicit_pos_{args.frames}.csv"
    vel_path=args.output_dir/f"implicit_vel_{args.frames}.csv"
    write_series(pos_path,pos)
    write_series(vel_path,vel)
    metric=equilibrium_metric(pos[-1],vel[-1])
    print(f"implicit {args.frames}-frame equilibrium metric")
    print(f"frames={args.frames}")
    print(f"mean_force={metric['mean_force']:.8g}")
    print(f"rms_force={metric['rms_force']:.8g}")
    print(f"max_force={metric['max_force']:.8g}")
    print(f"mean_vel={metric['mean_vel']:.8g}")
    print(f"max_vel={metric['max_vel']:.8g}")
    print(f"center_z={metric['center_z']:.8g}")
    print(f"saved={pos_path}")
    print(f"saved={vel_path}")


if __name__=="__main__":
    main()
