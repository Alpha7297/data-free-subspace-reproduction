from pathlib import Path

import argparse
import numpy as np
import scipy.optimize as opt

from config import *


ROOT=Path(__file__).resolve().parent
MAX_ITERS=10000
GRAD_THRESHOLD=1e-6
Z_FORCE=3.0
FORCE_IDS=[(HEIGHT//2)*WIDTH+(WIDTH//2)]
FREE_IDS=[i for i in range(HEIGHT*WIDTH) if i not in fixed_id]


def init_pos():
    i=np.arange(HEIGHT,dtype=np.float64)
    j=np.arange(WIDTH,dtype=np.float64)
    I,J=np.meshgrid(i,j,indexing="ij")
    K=np.zeros_like(I)
    return np.stack((I.T,J.T,K),axis=-1).reshape(-1,3)


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


EXTERNAL_FORCE=external_force()


def unpack(x):
    pos=init_pos()
    pos[FREE_IDS]=x.reshape(len(FREE_IDS),3)
    return pos


def energy_and_grad(x):
    pos=unpack(x)
    energy=-float(np.sum(EXTERNAL_FORCE*pos))
    force=EXTERNAL_FORCE.copy()
    for a,b in SPRING_PAIRS:
        l=pos[a]-pos[b]
        norm=np.sqrt(np.dot(l,l)+1e-12)
        ratio=(norm-origin_len)/norm
        energy+=0.5*k_hook*(norm-origin_len)**2
        spring_force=k_hook*ratio*l
        force[a]-=spring_force
        force[b]+=spring_force
    grad=-force[FREE_IDS].reshape(-1)
    return energy,grad


def residual_metric(pos):
    _,grad=energy_and_grad(pos[FREE_IDS].reshape(-1))
    residual=np.linalg.norm(grad.reshape(len(FREE_IDS),3),axis=-1)
    return {
        "mean_residual":float(np.mean(residual)),
        "rms_residual":float(np.sqrt(np.mean(residual*residual))),
        "max_residual":float(np.max(residual)),
        "center_z":float(pos[(HEIGHT//2)*WIDTH+(WIDTH//2),2]),
    }


def write_state(path,pos):
    with open(path,"w",encoding="utf-8") as f:
        for p in pos:
            f.write(f"{p[0]} {p[1]} {p[2]} ")
        f.write("\n")


def solve(max_iters=MAX_ITERS,grad_threshold=GRAD_THRESHOLD):
    x0=init_pos()[FREE_IDS].reshape(-1)
    result=opt.minimize(
        energy_and_grad,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter":max_iters,
            "gtol":grad_threshold,
            "ftol":1e-12,
            "maxls":50,
        },
    )
    return unpack(result.x),result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--max-iters",type=int,default=MAX_ITERS)
    parser.add_argument("--grad-threshold",type=float,default=GRAD_THRESHOLD)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"experiment")
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    pos,result=solve(args.max_iters,args.grad_threshold)
    path=args.output_dir/"static_pos.csv"
    write_state(path,pos)
    metric=residual_metric(pos)
    print("static equilibrium metric")
    print(f"success={result.success}")
    print(f"iters={result.nit}")
    print(f"energy={result.fun:.8g}")
    print(f"mean_residual={metric['mean_residual']:.8g}")
    print(f"rms_residual={metric['rms_residual']:.8g}")
    print(f"max_residual={metric['max_residual']:.8g}")
    print(f"center_z={metric['center_z']:.8g}")
    print(f"saved={path}")


if __name__=="__main__":
    main()
