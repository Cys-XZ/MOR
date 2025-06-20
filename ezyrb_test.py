import numpy as np
import matplotlib.pyplot as plt
from ezyrb import POD, RBF, Database, GPR, ANN, KNeighborsRegressor, RadiusNeighborsRegressor,  PODAE,AE
from ezyrb import ReducedOrderModel as ROM
from sklearn.gaussian_process.kernels import RBF as RBFS, WhiteKernel, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import warnings
import time
import tracemalloc

#导入数据
snapshots = np.load('D:/pycharm/PycharmProjects/streamlit/displacement_data/snapshots_z.npy')
param = np.load('D:/pycharm/PycharmProjects/streamlit/displacement_data/param.npy')


#单点测试
  # 指定验证数据的索引（从0开始）
validation_idx = 3
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
rbf = RBF(kernel='multiquadric', epsilon=0.02)
rom1 = ROM(db1, pod, rbf)
rom1.fit()

# 使用验证参数进行预测
result_db = rom1.predict([validation_param])
predicted_snapshot = result_db.snapshots_matrix
predicted_mean = np.mean(predicted_snapshot, axis=1)

# 创建散点图
plt.figure(figsize=(10, 6))
plt.scatter(training_params, snapshot_means, c='blue', alpha=0.6, label='Training data')
plt.scatter(validation_param, predicted_mean, c='red', alpha=0.4, label='Prediction')
plt.scatter(validation_param, validation_mean, c='green', alpha=0.4, label='Validation')
plt.xlabel('T', fontsize=16)
plt.ylabel('mean', fontsize=16)
plt.title('POD predicte_stress-RBF(multiquadric)', fontsize=20)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=14, loc='best', bbox_to_anchor=None)
plt.show()

# 随机选择一个点进行对比
random_idx = np.random.randint(0, len(validation_snapshot))
random_real = validation_snapshot[random_idx]
random_pred = predicted_snapshot.flatten()[random_idx]

plt.figure(figsize=(10, 6))
plt.scatter(training_params, snapshot_means, c='blue', alpha=0.6, label='Training data')
plt.scatter(validation_param, random_pred, c='red', alpha=0.4, label=f'Random Point Prediction (idx:{random_idx})')
plt.scatter(validation_param, random_real, c='green', alpha=0.4, label=f'Random Point Validation')
plt.xlabel('T', fontsize=16)
plt.ylabel('Value', fontsize=16)
plt.title('Random Point Comparison', fontsize=20)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=14, loc='best')
plt.show()

# 找出误差最大的点
error = np.abs(validation_snapshot - predicted_snapshot.flatten())
max_error_idx = np.argmax(error)
max_error_real = validation_snapshot[max_error_idx]
max_error_pred = predicted_snapshot.flatten()[max_error_idx]
relative_error = error[max_error_idx]/np.abs(validation_snapshot[max_error_idx])*100

plt.figure(figsize=(10, 6))
plt.scatter(training_params, snapshot_means, c='blue', alpha=0.6, label='Training data')
plt.scatter(validation_param, max_error_pred, c='red', alpha=0.4, label=f'Max Error Point Prediction (idx:{max_error_idx})')
plt.scatter(validation_param, max_error_real, c='green', alpha=0.4, label=f'Max Error Point Validation')
plt.xlabel('T', fontsize=16)
plt.ylabel('Value', fontsize=16)
plt.title(f'Maximum Error Point Comparison\nRelative Error: {relative_error:.2f}%', fontsize=20)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=14, loc='best')
plt.show()

db2 = Database(param, snapshots)
pod = POD()
rbf = RBF(kernel='multiquadric', epsilon=0.02)
rom2 = ROM(db2, pod, rbf)
rom2.fit()
errors = rom2.kfold_cv_error(n_splits =7)
print(errors)

# 创建柱状图显示交叉验证误差
plt.figure(figsize=(10, 6))
x_positions = np.arange(len(errors)) + 1  # 从1开始标记折数
plt.bar(x_positions, errors, color='skyblue', alpha=0.7)
plt.axhline(y=np.mean(errors), color='red', linestyle='--', label=f'Mean Error: {np.mean(errors):.2e}')

# 在每个柱子上标注具体的误差值
for i, error in enumerate(errors):
    plt.text(i+1, error, f'{error:.2e}', ha='center', va='bottom', fontsize=10)

plt.xlabel('Fold Number', fontsize=16)
plt.ylabel('Error', fontsize=16)
plt.title('K-fold Cross Validation Errors-RBF(multiquadric)', fontsize=20)
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend(fontsize=14, loc='best')
plt.xticks(x_positions)
plt.show()

# 添加真实数据和预测数据的点对点对比散点图
plt.figure(figsize=(12, 6))
x_indices = np.arange(len(validation_snapshot))
plt.scatter(x_indices, validation_snapshot, c='green', alpha=0.4, label='Real Data')
plt.scatter(x_indices, predicted_snapshot.flatten(), c='red', alpha=0.4, label='Predicted Data')
plt.xlabel('Data Point Index', fontsize=16)
plt.ylabel('Value', fontsize=16)
plt.title(f'Point-by-Point Comparison (Group {validation_idx + 1})', fontsize=20)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=14, loc='best', bbox_to_anchor=None)
plt.show()