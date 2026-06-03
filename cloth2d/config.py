import numpy as np
import torch
HEIGHT=20
WIDTH=20
k_hook=20.0
in_dim=20
origin_len=1.0
out_dim=1200-12
hid_dim=64
fixed_pos=[[0.0,0.0],[0.0,WIDTH*origin_len],
           [HEIGHT*origin_len,0.0],
           [HEIGHT*origin_len,WIDTH*origin_len]]
fixed_id=[0,WIDTH-1,WIDTH*(HEIGHT-1),WIDTH*HEIGHT-1]