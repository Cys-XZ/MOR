import pyvista as pv
import numpy as np
import pandas as pd
import os


def extract_displacement_components(file_path, deltaT="-50", output_dir=None, visualize=False):
    """
    从VTU文件中提取指定deltaT值的X、Y、Z位移分量数据

    参数:
    file_path: VTU文件的路径
    deltaT: 要提取的deltaT值，默认为"-50"
    output_dir: 输出目录，默认为None（不保存数据）
    visualize: 是否进行可视化，默认为False

    返回:
    x_data, y_data, z_data: 三个位移分量的numpy数组
    stress: 应力数据的numpy数组
    mesh: PyVista网格对象
    found_components: 包含找到的数据数组名称的字典
    """
    # 读取VTU文件
    try:
        mesh = pv.read(file_path)
        print(f"成功读取文件: {file_path}")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None, None, None, None, {}

    # 查找指定deltaT的X、Y、Z分量数据
    x_array_name = None
    y_array_name = None
    z_array_name = None
    s_array_name = None

    # 存储找到的数组名称
    found_components = {}

    # 首先尝试查找英文格式的
    for array_name in mesh.array_names:
        # 适应英文表头格式
        if "Displacement_field,_X-component" in array_name and f"deltaT={deltaT}" in array_name:
            x_array_name = array_name
            found_components['X'] = array_name
        elif "Displacement_field,_Y-component" in array_name and f"deltaT={deltaT}" in array_name:
            y_array_name = array_name
            found_components['Y'] = array_name
        elif "Displacement_field,_Z-component" in array_name and f"deltaT={deltaT}" in array_name:
            z_array_name = array_name
            found_components['Z'] = array_name
        elif "von_Mises_stress" in array_name and f"deltaT={deltaT}" in array_name:
            s_array_name = array_name
            found_components['S'] = array_name

    # 如果没有找到英文格式的，尝试查找中文格式的
    if x_array_name is None and y_array_name is None and z_array_name is None:
        for array_name in mesh.array_names:
            if f"X_分量" in array_name and f"deltaT={deltaT}" in array_name:
                x_array_name = array_name
                found_components['X'] = array_name
            elif f"Y_分量" in array_name and f"deltaT={deltaT}" in array_name:
                y_array_name = array_name
                found_components['Y'] = array_name
            elif f"Z_分量" in array_name and f"deltaT={deltaT}" in array_name:
                z_array_name = array_name
                found_components['Z'] = array_name
            elif f"S_分量" in array_name and f"deltaT={deltaT}" in array_name:
                s_array_name = array_name
                found_components['S'] = array_name

    # 输出找到的分量标识
    if found_components:
        print(f"\n找到的位移分量 (deltaT={deltaT}):")
        for component, name in found_components.items():
            print(f"{component}分量: {name}")
    else:
        print(f"\n未找到deltaT={deltaT}的位移分量数据")

    # 提取各分量数据
    x_data = mesh.get_array(x_array_name) if x_array_name else None
    y_data = mesh.get_array(y_array_name) if y_array_name else None
    z_data = mesh.get_array(z_array_name) if z_array_name else None
    stress = mesh.get_array(s_array_name) if s_array_name else None

    # 如果设置了输出目录，保存数据到CSV文件
    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"创建输出目录: {output_dir}")

        # 准备坐标数据
        points = mesh.points

        # 创建DataFrame
        data = {'X坐标': points[:, 0], 'Y坐标': points[:, 1], 'Z坐标': points[:, 2]}

        if x_data is not None:
            data['X位移'] = x_data
        if y_data is not None:
            data['Y位移'] = y_data
        if z_data is not None:
            data['Z位移'] = z_data

        # 计算合位移
        if x_data is not None and y_data is not None and z_data is not None:
            data['合位移'] = np.sqrt(x_data ** 2 + y_data ** 2 + z_data ** 2)

        # 保存为CSV
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        csv_path = os.path.join(output_dir, f"{file_name}_deltaT{deltaT}_displacement.csv")
        pd.DataFrame(data).to_csv(csv_path, index=False)
        print(f"数据已保存至: {csv_path}")

    # 如果需要可视化
    if visualize and x_data is not None and y_data is not None and z_data is not None:
        visualize_displacement(mesh, x_data, y_data, z_data, deltaT)

    return x_data, y_data, z_data, stress, mesh, found_components


def calculate_magnitude(x_data, y_data, z_data):
    """
    计算位移的合成大小

    参数:
    x_data, y_data, z_data: 三个位移分量的numpy数组

    返回:
    magnitude: 位移合成大小的numpy数组
    """
    if x_data is None or y_data is None or z_data is None:
        print("无法计算合位移，缺少分量数据")
        return None

    magnitude = np.sqrt(x_data ** 2 + y_data ** 2 + z_data ** 2)
    return magnitude


def visualize_displacement(mesh, x_data, y_data, z_data, deltaT, save_path=None):
    """
    可视化位移场数据

    参数:
    mesh: PyVista网格对象
    x_data, y_data, z_data: 三个位移分量的numpy数组
    deltaT: 温度变化值，用于标题显示
    save_path: 保存图像的路径，如果为None则不保存
    """
    if x_data is None or y_data is None or z_data is None:
        print("缺少位移分量数据，无法可视化")
        return

    # 计算位移大小
    magnitude = calculate_magnitude(x_data, y_data, z_data)

    # 添加位移大小和向量场到网格
    mesh.point_data["位移大小"] = magnitude
    mesh.point_data["位移向量"] = np.column_stack((x_data, y_data, z_data))

    # 设置中文字体支持
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建可视化窗口
    p = pv.Plotter(shape=(1, 2), window_size=[1600, 800])

    # 第一个子图：显示位移大小
    p.subplot(0, 0)
    p.add_text(f"位移大小 (deltaT={deltaT})", font_size=14)
    p.add_mesh(mesh, scalars="位移大小", cmap="rainbow",
               show_edges=False, scalar_bar_args={"title": "位移大小"})
    p.view_isometric()
    p.add_axes()

    # 第二个子图：显示位移向量场
    p.subplot(0, 1)
    p.add_text(f"位移向量场 (deltaT={deltaT})", font_size=14)

    # 显示网格
    p.add_mesh(mesh, color="lightgray", opacity=0.3, show_edges=False)

    # 计算合适的箭头比例
    mesh_size = max(mesh.bounds[1] - mesh.bounds[0],
                    mesh.bounds[3] - mesh.bounds[2],
                    mesh.bounds[5] - mesh.bounds[4])
    arrow_scale = mesh_size / 20

    # 为了避免箭头过密，可以抽样显示
    # 创建均匀的点采样
    sampling = max(1, mesh.n_points // 500)  # 最多显示约500个箭头

    # 添加箭头
    p.add_arrows(mesh.points[::sampling],
                 np.column_stack((x_data, y_data, z_data))[::sampling],
                 mag=arrow_scale,
                 name="arrows",
                 color="red")

    p.view_isometric()
    p.add_axes()

    # 链接两个视图的相机
    p.link_views()

    # 保存图像
    if save_path:
        p.screenshot(save_path, return_img=False)
        print(f"图像已保存至: {save_path}")

    # 显示可视化窗口
    p.show()


def list_available_deltats(file_path):
    """
    列出VTU文件中所有可用的deltaT值

    参数:
    file_path: VTU文件的路径

    返回:
    delta_t_values: 所有可用的deltaT值列表
    """
    try:
        mesh = pv.read(file_path)
        print(f"成功读取文件: {file_path}")
    except Exception as e:
        print(f"读取文件失败: {e}")
        return []

    # 提取所有deltaT值
    delta_t_values = set()

    # 正则表达式匹配deltaT值
    import re
    pattern1 = r'deltaT=(-?\d+)'  # 匹配deltaT=数字
    pattern2 = r'deltaT=(-?\d+\.\d+)'  # 匹配deltaT=小数

    for array_name in mesh.array_names:
        # 尝试匹配
        match1 = re.search(pattern1, array_name)
        match2 = re.search(pattern2, array_name)

        if match1:
            delta_t_values.add(match1.group(1))
        elif match2:
            delta_t_values.add(match2.group(1))

    delta_t_values = sorted(list(delta_t_values), key=lambda x: float(x))

    if delta_t_values:
        print(f"\n可用的deltaT值:")
        for dt in delta_t_values:
            print(f"  - {dt}")
    else:
        print("未找到任何deltaT值")

    return delta_t_values


if __name__ == "__main__":
    # 示例用法
    file_path = r"C:/Users/chenyushi/Desktop/data/L_deltaT.vtu"  # 替换为实际文件路径

    # 列出所有可用的deltaT值
    deltats = list_available_deltats(file_path)

    if deltats:
        # 选择一个deltaT值进行提取
        deltaT = deltats[0]  # 选择第一个可用的deltaT

        # 提取位移分量，不保存数据，但进行可视化
        x_data, y_data, z_data, mesh, found_components = extract_displacement_components(
            file_path,
            deltaT=deltaT,
            output_dir=None,  # 不保存数据
            visualize=False  # 进行可视化
        )
