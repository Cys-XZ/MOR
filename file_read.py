import numpy as np
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
import math
from ezyrb import POD, RBF, Database,AE,PODAE
from ezyrb import ReducedOrderModel as ROM
from ezyrb import RBF, GPR, KNeighborsRegressor, RadiusNeighborsRegressor, Linear, ANN
import torch.nn as nn
from sklearn.model_selection import KFold
import time
import pyvista as pv
from extract_displacement_components import list_available_deltats,extract_displacement_components,visualize_displacement

file_path = r"C:/Users/chenyushi/Desktop/data/2025_0606_CoWoS_m1_meshdata1.vtu"  # 替换为实际文件路径
snapshots_x=[]
snapshots_y=[]
snapshots_z=[]
snapshots_stress=[]
# 列出所有可用的deltaT值
deltats = list_available_deltats(file_path)
for deltaT in deltats:
    x_data, y_data, z_data,stress, mesh, found_components = extract_displacement_components(
        file_path,
        deltaT=deltaT,
        output_dir=None,  # 不保存数据
        visualize=False  # 进行可视化
    )
    snapshots_x.append(x_data)
    snapshots_y.append(y_data)
    snapshots_z.append(z_data)
    snapshots_stress.append(stress)

snapshots_x=np.array(snapshots_x)
snapshots_y=np.array(snapshots_y)
snapshots_z=np.array(snapshots_z)
snapshots_stress=np.array(snapshots_stress)

param=(np.array(range(-50,90,20)).T)
param=param.reshape(-1,1)