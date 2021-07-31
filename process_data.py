import os
import numpy as np
import pandas as pd


data_dir = '/Users/wangshuo/Documents/workspace/2019-ZJU_SummerResearch/data'
npz_fp = './data/data_dict.npz'

data = pd.read_csv(os.path.join(data_dir, 'data.csv'), header=None)
w_adj = pd.read_csv(os.path.join(data_dir, 'weight_adj.csv'), header=None)
w_dis = pd.read_csv(os.path.join(data_dir, 'weight_dis.csv'), header=None)
w_simi = pd.read_csv(os.path.join(data_dir, 'weight_simi.csv'), header=None)

np.savez_compressed(
    npz_fp,
    taxi=data,
    neighbor_adj=w_adj,
    trans_adj=w_dis,
    semantic_adj=w_simi,
)
