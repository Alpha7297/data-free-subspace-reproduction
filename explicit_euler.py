import numpy as np
LENGTH=100
k_hook=2.0
leng_origin=1.0
pos_Q=np.array([[i,0] for i in range(LENGTH)],dtype=np.float32)
vel_Q=np.zeros_like(pos_Q)
for i in range(LENGTH//2):
    vel_Q[i+LENGTH//2]=[0,-1]
totaldt=0.01
frames=500
with open("experiment/explicit_pos.csv",'w') as f1:
    with open("experiment/explicit_vel.csv",'w') as f2:

        for frame in range(frames):
            for i in range(LENGTH):
                f1.write(f"{pos_Q[i][0]} {pos_Q[i][1]} ")
                f2.write(f"{vel_Q[i][0]} {vel_Q[i][1]} ")
            f1.write("\n")
            f2.write("\n")
            dt=totaldt/1000.0
            for _ in range(1000):
                acc_Q=np.zeros_like(pos_Q)
                for i in range(LENGTH-1):
                    l=pos_Q[i]-pos_Q[i+1]
                    norm=np.sqrt(np.dot(l,l))+1e-5
                    force=k_hook*(norm-leng_origin)*l/norm
                    acc_Q[i]-=force
                    acc_Q[i+1]+=force
                vel_Q=acc_Q*dt+vel_Q
                pos_Q=pos_Q+vel_Q*dt
                vel_Q[0]=np.array([0,0])
                pos_Q[0]=np.array([0,0])
