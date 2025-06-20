import numpy as np
import matplotlib.pyplot as plt
from ezyrb import POD, RBF, Database, GPR, ANN, KNeighborsRegressor, RadiusNeighborsRegressor,  PODAE, AE
from ezyrb import ReducedOrderModel as ROM
from sklearn.gaussian_process.kernels import RBF as RBFS, WhiteKernel, ConstantKernel, Matern
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import warnings
import time
import tracemalloc


warnings.filterwarnings('ignore')

# 加载数据


snapshots1 = np.load(
    'D:/pycharm/PycharmProjects/EZyRB-master/EZyRB-master/data/2025_0606_CoWoS_m1_meshdata1/snapshots_stress.npy')
param = np.load('D:/pycharm/PycharmProjects/EZyRB-master/EZyRB-master/data/2025_0606_CoWoS_m1_meshdata1/param.npy')
print(np.mean(snapshots1))
print(np.std(snapshots1))
db1 = Database(param, snapshots1)

# 准备测试参数 - 用于预测性能测试
n_samples = len(param)
print(snapshots1.shape)
n_test = max(1, int(n_samples * 0.2))  # 使用20%的数据作为测试集

# 确保处理单数样本数的情况
if n_samples % 2 != 0:
    train_indices = np.arange(0, n_samples - 1, 1)  # 排除最后一个样本确保偶数
    test_indices = np.array([n_samples - 1])  # 最后一个样本作为测试集
else:
    # 随机选择测试集
    test_indices = np.random.choice(n_samples, n_test, replace=False)
    train_indices = np.array([i for i in range(n_samples) if i not in test_indices])

# 创建训练和测试数据库
train_db = Database(param[train_indices], snapshots1[train_indices])
test_params = param[test_indices]
test_snapshots = snapshots1[test_indices]
test_db = Database(test_params, test_snapshots)
kernel_rbf = ConstantKernel(1.0, constant_value_bounds=[1e-5, 1e8]) * RBFS(
    length_scale=1, length_scale_bounds=[1e-5, 1e8])

kernel_Matern = ConstantKernel(1.0, constant_value_bounds=[1e-5, 1e8]) * Matern(
    length_scale=1, nu=1.5, length_scale_bounds=[1e-5, 1e8])

kernel_White = ConstantKernel(1.0, constant_value_bounds=[1e-15, 1e5]) * RBFS(
    length_scale=1, length_scale_bounds=[1e-15, 1e15]) + WhiteKernel(
    noise_level=1e-3, noise_level_bounds=(1e-15, 1e15))
# 降维方法
reductions = {
    'POD_SVD': POD('svd'),
    # 'kpca':KPCA(kernel='rbf', gamma=20, rank=-1)
}
'''
    'RBF':GPR(kern=kernel_rbf,normalizer=False, optimization_restart=60),
    'Matern':GPR(kern=kernel_Matern,normalizer=False, optimization_restart=45),
    'White':GPR(kernel_White,normalizer=False, optimization_restart=45)
    'linear': RBF(kernel='linear'),
    'gaussian': RBF(kernel='gaussian', epsilon=0.005),
    'spline': RBF(kernel='thin_plate_spline'),
    'i_multiquadric': RBF(kernel='inverse_multiquadric', epsilon=0.003),
    'multiquadric': RBF(kernel='multiquadric', epsilon=0.005),
    'cubic': RBF(kernel='cubic'),
    'quintic': RBF(kernel='quintic')
    'Tanh': ANN([4,16,32], function=nn.Tanh(),stop_training=[10000, 1e-12],lr=1e-4,l2_regularization=1e-4,frequency_print=1000),
    'ReLU': ANN([4,16,32], function=nn.ReLU(), stop_training=[10000, 1e-12], lr=1e-4,l2_regularization=1e-4,frequency_print=1000),
    'PReLU': ANN([4,16,32], function=nn.PReLU(),stop_training=[10000, 1e-12],lr=1e-4,l2_regularization=1e-4,frequency_print=1000),
    'Sigmoid': ANN([4,16,32], function=nn.Sigmoid(),stop_training=[10000, 1e-12],lr=1e-4,l2_regularization=1e-4,frequency_print=1000),
'Softmax': ANN([4,16,32], function=nn.Softmax(),stop_training=[10000, 1e-12],lr=1e-4,l2_regularization=1e-4,frequency_print=1000),
'LeakyReLU': ANN([4,16,32], function=nn.LeakyReLU(),stop_training=[10000, 1e-12],lr=1e-4,l2_regularization=1e-4,frequency_print=1000),
'ELU': ANN([4,16,32], function=nn.ELU(),stop_training=[10000, 1e-12],lr=1e-4,l2_regularization=1e-4,frequency_print=1000),'''
# 近似方法
approximations = {

    'RBF': RBF(kernel='multiquadric', epsilon=0.02),
    'GPR': GPR(kern=kernel_Matern, normalizer=False, optimization_restart=250),
    'KN': KNeighborsRegressor(n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30, metric='minkowski'),
    'RN': RadiusNeighborsRegressor(radius=60, weights='distance', algorithm='auto', leaf_size=30, metric='minkowski'),
    'ANN': ANN([6, 12, 24], function=nn.ReLU(), stop_training=[10000, 1e-12], lr=1e-1, l2_regularization=1e-2,
               frequency_print=1000), }

# 存储性能数据
performance_data = {
    'errors': {},  # 使用K折交叉验证的误差
    'prediction_times': {},  # 仅预测阶段的时间
    'prediction_memories': {},  # 仅预测阶段的内存
    'fit_times': {}  # 可选：记录训练时间
}

# 设置列宽
col_width = 18
first_col_width = 25

# 第一部分：使用K折交叉验证测试误差
for redname, redclass in reductions.items():
    performance_data['errors'][redname] = {}

    for approxname, approxclass in approximations.items():
        try:
            # 使用EZyRB自带的K折交叉验证评估误差
            rom = ROM(db1, redclass, approxclass)
            errors = rom.kfold_cv_error(n_splits=7)  # 使用4折交叉验证
            avg_error = errors.mean()

            # 存储结果
            performance_data['errors'][redname][approxname] = avg_error

        except Exception as e:
            print(f"错误: {redname} + {approxname} K折交叉验证 - {str(e)}")

# 第二部分：测试预测阶段的时间和内存使用
for redname, redclass in reductions.items():
    performance_data['prediction_times'][redname] = {}
    performance_data['prediction_memories'][redname] = {}
    performance_data['fit_times'][redname] = {}

    for approxname, approxclass in approximations.items():
        try:
            # 1. 先训练模型（不计入性能测试）
            fit_start_time = time.time()
            rom = ROM(train_db, redclass, approxclass)
            rom.fit()
            fit_time = time.time() - fit_start_time
            performance_data['fit_times'][redname][approxname] = fit_time

            # 2. 仅对预测阶段进行性能测试
            all_pred_times = []
            all_pred_memories = []

            for i, test_param in enumerate(test_params):
                # 开始内存追踪
                tracemalloc.start()

                # 记录预测开始时间
                pred_start_time = time.time()

                # 执行预测
                pred_result = rom.predict([test_param])

                # 记录预测结束时间
                pred_time = time.time() - pred_start_time

                # 获取内存峰值
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                peak_memory_mb = peak / 1024 / 1024

                all_pred_times.append(pred_time)
                all_pred_memories.append(peak_memory_mb)

            # 计算平均值
            avg_pred_time = np.mean(all_pred_times)
            avg_pred_memory = np.mean(all_pred_memories)

            # 存储结果
            performance_data['prediction_times'][redname][approxname] = avg_pred_time
            performance_data['prediction_memories'][redname][approxname] = avg_pred_memory

        except Exception as e:
            print(f"错误: {redname} + {approxname} 预测测试 - {str(e)}")

# 计算近似方法平均值
approx_avg_data = {}
for approxname in approximations:
    approx_avg_data[approxname] = {
        'pred_times': [],
        'memories': [],
        'errors': []
    }

    for redname in reductions:
        if redname in performance_data['prediction_times'] and approxname in performance_data['prediction_times'][
            redname]:
            approx_avg_data[approxname]['pred_times'].append(performance_data['prediction_times'][redname][approxname])
            approx_avg_data[approxname]['memories'].append(performance_data['prediction_memories'][redname][approxname])
            approx_avg_data[approxname]['errors'].append(performance_data['errors'][redname][approxname])

    # 计算平均值
    if approx_avg_data[approxname]['pred_times']:
        approx_avg_data[approxname]['avg_pred_time'] = np.mean(approx_avg_data[approxname]['pred_times'])
        approx_avg_data[approxname]['avg_memory'] = np.mean(approx_avg_data[approxname]['memories'])
        approx_avg_data[approxname]['avg_error'] = np.mean(approx_avg_data[approxname]['errors'])

# =============== 开始打印所有结果 ===============

# 打印表头
header = '{:<{}s}'.format('', first_col_width)
for name in approximations:
    header += '{:>{}s}'.format(name, col_width)

# 打印K折交叉验证误差结果
print("\nK折交叉验证误差结果:")
print(header)
print('-' * (first_col_width + len(approximations) * col_width))

for redname in reductions:
    row = '{:<{}s}'.format(redname, first_col_width)
    for approxname in approximations:
        if approxname in performance_data['errors'].get(redname, {}):
            row += '{:>{}.4e}'.format(performance_data['errors'][redname][approxname], col_width)
        else:
            row += '{:>{}s}'.format('N/A', col_width)
    print(row)

# 打印预测时间统计
print('\n\n预测执行时间 (秒):')
print(header)
print('-' * (first_col_width + len(approximations) * col_width))
for redname in reductions:
    row = '{:<{}s}'.format(redname, first_col_width)
    for approxname in approximations:
        if approxname in performance_data['prediction_times'].get(redname, {}):
            row += '{:>{}.6f}'.format(performance_data['prediction_times'][redname][approxname], col_width)
        else:
            row += '{:>{}s}'.format('N/A', col_width)
    print(row)

# 打印内存统计
print('\n\n预测内存峰值 (MB):')
print(header)
print('-' * (first_col_width + len(approximations) * col_width))
for redname in reductions:
    row = '{:<{}s}'.format(redname, first_col_width)
    for approxname in approximations:
        if approxname in performance_data['prediction_memories'].get(redname, {}):
            row += '{:>{}.2f}'.format(performance_data['prediction_memories'][redname][approxname], col_width)
        else:
            row += '{:>{}s}'.format('N/A', col_width)
    print(row)

# 打印训练时间
print('\n\n训练执行时间 (秒) :')
print(header)
print('-' * (first_col_width + len(approximations) * col_width))
for redname in reductions:
    row = '{:<{}s}'.format(redname, first_col_width)
    for approxname in approximations:
        if approxname in performance_data['fit_times'].get(redname, {}):
            row += '{:>{}.3f}'.format(performance_data['fit_times'][redname][approxname], col_width)
        else:
            row += '{:>{}s}'.format('N/A', col_width)
    print(row)

# 汇总统计
print('\n汇总统计')
print('=' * 80)

# 按近似方法统计
print('\n按近似方法平均值:')
print('{:<25s} {:>15s} {:>15s} {:>15s}'.format('近似方法', '平均预测时间(s)', '平均内存(MB)', 'K折交叉验证误差'))
print('-' * 70)
for approxname in approximations:
    if 'avg_pred_time' in approx_avg_data.get(approxname, {}):
        print('{:<25s} {:>15.6f} {:>15.2f} {:>15.4e}'.format(
            approxname,
            approx_avg_data[approxname]['avg_pred_time'],
            approx_avg_data[approxname]['avg_memory'],
            approx_avg_data[approxname]['avg_error']
        ))

# 绘制可视化图表
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 准备数据
approx_names = []
approx_pred_times = []
approx_memories = []
approx_errors = []

for approxname in approximations:
    if 'avg_pred_time' in approx_avg_data.get(approxname, {}):
        approx_names.append(approxname)
        approx_pred_times.append(approx_avg_data[approxname]['avg_pred_time'])
        approx_memories.append(approx_avg_data[approxname]['avg_memory'])
        approx_errors.append(approx_avg_data[approxname]['avg_error'])

# 预测时间柱状图
fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(approx_names))
bars = ax.bar(x_pos, approx_pred_times, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
ax.set_xlabel('映射方法', fontsize=18)
ax.set_ylabel('平均预测时间 (秒)', fontsize=18)
ax.set_title('不同映射方法的平均预测时间', fontsize=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(approx_names, rotation=0, ha='right', fontsize=14)

# 在柱子上添加数值
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height,
            f'{height:.6f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('approximation_methods_prediction_time.png', dpi=300, bbox_inches='tight')
plt.show()

# 内存使用柱状图
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(x_pos, approx_memories, color='lightyellow', edgecolor='orange', alpha=0.7)
ax.set_xlabel('映射方法', fontsize=18)
ax.set_ylabel('平均预测内存使用 (MB)', fontsize=18)
ax.set_title('不同映射方法的平均预测内存使用', fontsize=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(approx_names, rotation=0, ha='right', fontsize=14)

# 在柱子上添加数值
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height,
            f'{height:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('approximation_methods_prediction_memory.png', dpi=300, bbox_inches='tight')
plt.show()

# K折交叉验证误差柱状图
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(x_pos, approx_errors, color='lightcoral', edgecolor='darkred', alpha=0.7)
ax.set_xlabel('映射方法', fontsize=18)
ax.set_ylabel('平均K折交叉验证误差', fontsize=18)
ax.set_title('不同核函数的K折交叉验证误差 (7折)', fontsize=20)
ax.set_xticks(x_pos)
ax.set_xticklabels(approx_names, rotation=0, ha='right', fontsize=14)
ax.set_yscale('log')  # 使用对数刻度显示误差

# 在柱子上添加数值
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., height,
            f'{height:.2e}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('approximation_methods_kfold_error.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n所有图表已保存到当前目录:")
print("- approximation_methods_prediction_time.png")
print("- approximation_methods_prediction_memory.png")
print("- approximation_methods_kfold_error.png")