import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
LENGTH=100
k_hook=2.0
leng_origin=1.0
pos_Q=np.array([[i,0] for i in range(LENGTH)],dtype=np.float32)
vel_Q=np.zeros_like(pos_Q)
for i in range(LENGTH//2):
    vel_Q[i+LENGTH//2]=[0,-1]
dt=0.01
frames=500

def free_id(i,d):
    return (i-1)*2+d

with open("experiment/implicit_pos.csv",'w') as f1:
    with open("experiment/implicit_vel.csv",'w') as f2:
        for frame in range(frames):
            for i in range(LENGTH):
                f1.write(f"{pos_Q[i][0]} {pos_Q[i][1]} ")
                f2.write(f"{vel_Q[i][0]} {vel_Q[i][1]} ")
            f1.write("\n")
            f2.write("\n")
            force_Q=np.zeros_like(pos_Q)
            rows=[]
            cols=[]
            data=[]
            b=np.zeros((LENGTH-1)*2,dtype=np.float32)

            for i in range(LENGTH-1):
                l=pos_Q[i]-pos_Q[i+1]
                norm=np.sqrt(np.dot(l,l))+1e-5
                a=(norm-leng_origin)/norm
                force=k_hook*a*l
                force_Q[i]-=force
                force_Q[i+1]+=force

                I=np.eye(2,dtype=np.float32)
                outer=np.outer(l,l)
                D=-k_hook*(a*I+leng_origin*outer/(norm**3))
                blocks=[(i,i,D),(i,i+1,-D),(i+1,i,-D),(i+1,i+1,D)]
                for bi,bj,block in blocks:
                    if bi==0 or bj==0:
                        continue
                    for r in range(2):
                        for c in range(2):
                            rows.append(free_id(bi,r))
                            cols.append(free_id(bj,c))
                            data.append(-dt*dt*block[r,c])

            for i in range(1,LENGTH):
                for d in range(2):
                    idx=free_id(i,d)
                    rows.append(idx)
                    cols.append(idx)
                    data.append(1.0)
                    b[idx]=dt*vel_Q[i,d]+dt*dt*force_Q[i,d]

            A=sp.coo_matrix((data,(rows,cols)),shape=((LENGTH-1)*2,(LENGTH-1)*2)).tocsr()
            dq=spla.spsolve(A,b).reshape(LENGTH-1,2)
            pos_Q[1:]+=dq
            vel_Q[1:]=dq/dt
            pos_Q[0]=np.array([0,0])
            vel_Q[0]=np.array([0,0])
