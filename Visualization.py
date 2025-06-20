import numpy as np
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
import math
from ezyrb import POD, RBF, Database,AE,PODAE
from ezyrb import ReducedOrderModel as ROM

from ezyrb import RBF, GPR, KNeighborsRegressor, RadiusNeighborsRegressor, Linear, ANN
import torch.nn as nn
from sklearn.model_selection import KFold
import pyvista as pv

#导入网格数据
mesh = pv.read("C:/Users/chenyushi/Desktop/data/2025_0606_CoWoS_m1_meshdata1.vtu")
points = mesh.points

#导入网格渲染数据
  # 查看所有点数据的名称
print("可用的点数据字段：")
for name in mesh.point_data.keys():
    print(f"  - {name}")

  # 查看所有单元数据的名称（如果有）
print("\n可用的单元数据字段：")
for name in mesh.cell_data.keys():
    print(f"  - {name}")
points = mesh.points # shape: (n_points, 3)
  # u 数据
u_name ="Displacement_field,_X-component_@_deltaT=70"
u_data = mesh.point_data[u_name] # shape: (n_points, )
mesh.point_data["u"]=u_data
scalars = mesh["u"]
#基本可视化
p = pv.Plotter()
p.add_mesh(mesh,scalars='u', opacity=0.3,cmap="plasma", show_edges=True)
p.view_xy()
p.show()

#变形数据
u=mesh.point_data["Displacement_field,_X-component_@_deltaT=70"]
v=mesh.point_data["Displacement_field,_Y-component_@_deltaT=70"]
w=mesh.point_data["Displacement_field,_Z-component_@_deltaT=70"]

displacement = np.stack([u, v, w], axis=1)
mesh["displacement"] = displacement

warped = mesh.copy()
warped = warped.warp_by_vector("displacement", factor=30)
p = pv.Plotter()
# 变形前后⽹格
p.add_mesh(mesh, color="gray", opacity=1, show_edges=True, label="original")
p.add_mesh(warped, scalars="u", opacity=0.3,cmap="plasma", show_edges=True, label="deformed")
p.add_legend()
p.show()

#预测误差图
   #导入数据
snapshots = np.load(
    'D:/pycharm/PycharmProjects/EZyRB-master/EZyRB-master/data/2025_0606_CoWoS_m1_meshdata1/snapshots_stress.npy')
param = np.load('D:/pycharm/PycharmProjects/EZyRB-master/EZyRB-master/data/2025_0606_CoWoS_m1_meshdata1/param.npy')

  # 指定验证数据的索引（从0开始）
validation_idx = 4  # 使用第3组数据作为验证数据

  # 获取验证数据
validation_param = param[validation_idx]
validation_snapshot = snapshots[validation_idx]
validation_mean = np.mean(validation_snapshot)

  # 构建训练数据集（除去验证数据）
training_indices = list(range(len(param)))
training_indices.remove(validation_idx)
training_params = param[training_indices]
training_snapshots = snapshots[training_indices]
snapshot_means = np.mean(training_snapshots, axis=1)

  # 构建数据库和降阶模型
db1 = Database(training_params, training_snapshots)
pod = POD()
rbf = RBF(kernel='quintic')
rom1 = ROM(db1, pod, rbf)
rom1.fit()
  # 3. 创建训练数据库
new_param = param[validation_idx]
result_db = rom1.predict([new_param])
predicted_snapshot = result_db.snapshots_matrix
predicted_snapshot=predicted_snapshot
  #  6. 获取对应的真实数据
true_snapshot = snapshots[validation_idx,:]
  #计算单点误差（一维原始数据可能有0点，分母取所有原始数据平均值）
mean_true_snapshot=np.mean(true_snapshot)
error = np.linalg.norm(predicted_snapshot - true_snapshot,axis=0)/mean_true_snapshot
  #设置待标记的点范围
std=np.std(error)
mean=np.mean(error)

above_threshold = error > (mean+1*std)

  # 创建正确的RGB颜色数组（四通道，包含透明度）
colors = np.zeros((mesh.n_points, 4))  # 形状：(n_points, 4) - RGBA
colors[above_threshold] = [1, 0, 0, 1]  # 阈值以上：红色，完全不透明
colors[~above_threshold] = [0, 0, 1, 0.01]  # 阈值以下：蓝色，半透明

  # === 预测误差图 ===
p = pv.Plotter()
p.add_mesh(
    mesh,
    scalars=colors,  # 传入RGBA数组
    rgba=True,  # 启用RGBA模式
    show_scalar_bar=False,  # 隐藏默认标量条
    name="threshold_mesh"  # 为后续交互添加名称
)
p.view_xy()
p.show()









