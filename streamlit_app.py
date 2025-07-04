import streamlit as st
import numpy as np
import os
import sys
import tempfile
from pathlib import Path
import pyvista as pv
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from extract_displacement_components import list_available_deltats, extract_displacement_components, visualize_displacement
import torch.nn as nn

# 检测是否在云环境中运行
def is_cloud_environment():
    """检测是否在云环境中运行（没有图形界面）"""
    try:
        # 检查是否在常见的云平台上
        cloud_indicators = [
            'STREAMLIT_SHARING', 'STREAMLIT_CLOUD', 'STREAMLIT_SERVER_HEADLESS',
            'HEROKU', 'RAILWAY', 'RENDER', 'VERCEL', 'REPLIT_CLUSTER',
            'CODESPACE_NAME', 'GITPOD_WORKSPACE_ID'
        ]
        if any(indicator in os.environ for indicator in cloud_indicators):
            return True
        
        # 只在Linux系统上检查DISPLAY环境变量
        if os.name == 'posix' and 'DISPLAY' not in os.environ:
            # 进一步检查是否真的在无头环境中
            try:
                import subprocess
                # 检查是否可以运行图形相关命令
                result = subprocess.run(['which', 'Xvfb'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    return False  # 有Xvfb，可能是开发环境
                
                # 检查是否在SSH会话中
                if 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ:
                    return True
                    
                return True  # Linux系统没有DISPLAY且不在SSH中，可能是云环境
            except:
                return True
        
        # Windows和macOS通常不需要DISPLAY环境变量
        if os.name in ['nt', 'posix'] and sys.platform == 'darwin':
            return False
            
        return False
    except:
        return False  # 出错时假设不是云环境

# 配置PyVista用于云环境和中文支持
def configure_pyvista_for_cloud():
    """为云环境配置PyVista并设置中文支持"""
    is_cloud = is_cloud_environment()
    
    # 设置中文字体支持
    try:
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        
        # 设置matplotlib中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 设置PyVista的matplotlib后端
        pv.global_theme.font.family = 'arial'
        pv.global_theme.font.size = 12
        
    except Exception as e:
        print(f"字体配置警告: {str(e)}")
    
    if is_cloud:
        try:
            # 设置PyVista为离屏模式
            pv.OFF_SCREEN = True
            
            # 设置环境变量（云端无头环境）
            os.environ['PYVISTA_OFF_SCREEN'] = 'true'
            os.environ['PYVISTA_USE_PANEL'] = 'false'
            os.environ['PYVISTA_JUPYTER_BACKEND'] = 'static'
            
            # 禁用GPU相关功能
            os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
            os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
            
            # 设置X11相关环境变量（避免libXrender错误）
            os.environ['QT_QPA_PLATFORM'] = 'offscreen'
            os.environ['DISPLAY'] = ':99'
            os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'
            os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
            
            # 禁用Qt相关的图形功能
            os.environ['QT_QPA_FONTDIR'] = '/usr/share/fonts'
            os.environ['MPLBACKEND'] = 'Agg'
            
            # 尝试启动虚拟显示器（如果可用）
            try:
                # 首先尝试使用xvfbwrapper
                try:
                    from xvfbwrapper import Xvfb
                    vdisplay = Xvfb(width=1280, height=720, colordepth=24)
                    vdisplay.start()
                    print("✅ Xvfb虚拟显示器启动成功")
                except ImportError:
                    print("⚠️ xvfbwrapper未安装，尝试直接启动Xvfb")
                    # 尝试直接启动Xvfb
                    import subprocess
                    subprocess.Popen(['Xvfb', ':99', '-screen', '0', '1280x720x24'], 
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print("✅ 直接启动Xvfb成功")
                except Exception as e:
                    print(f"⚠️ 无法启动Xvfb: {e}")
                
                # 然后尝试PyVista的xvfb
                pv.start_xvfb()
                print("✅ PyVista Xvfb启动成功")
            except Exception as e:
                print(f"⚠️ 虚拟显示器启动失败: {e}")
                # 即使Xvfb失败，也继续运行（使用纯软件渲染）
                
        except Exception as e:
            # 在Streamlit中显示警告（如果可用）
            try:
                st.warning(f"PyVista云环境配置警告: {str(e)}")
            except:
                print(f"PyVista云环境配置警告: {str(e)}")
    else:
        # 本地环境配置
        try:
            # 确保离屏模式关闭（本地交互式使用）
            pv.OFF_SCREEN = False
            # 设置默认主题
            pv.set_plot_theme("document")
            
        except Exception as e:
            print(f"PyVista本地环境配置警告: {str(e)}")
    
    # 调试信息（可选）
    # print(f"云环境检测结果: {is_cloud}")
    # print(f"PyVista OFF_SCREEN: {pv.OFF_SCREEN}")

# 创建云环境友好的3D可视化函数
def create_cloud_friendly_plot(mesh, **kwargs):
    """创建云环境友好的3D图像"""
    try:
        # 方法1: 使用PyVista离屏渲染
        return create_pyvista_plot(mesh, **kwargs)
    except Exception as e1:
        try:
            # 方法2: 使用matplotlib 3D替代
            return create_matplotlib_3d_plot(mesh, **kwargs)
        except Exception as e2:
            # 方法3: 使用2D投影图
            return create_2d_projection_plot(mesh, **kwargs)

def create_pyvista_plot(mesh, scalars=None, cmap='viridis', opacity=0.8, show_edges=True, title="3D Visualization"):
    """使用PyVista创建3D图像"""
    import pyvista as pv
    
    # 强制设置离屏模式
    pv.OFF_SCREEN = True
    
    try:
        # 在云端环境中设置额外的安全措施
        if is_cloud_environment():
            # 确保使用软件渲染
            os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
            os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
            
        # 创建绘图器
        plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
        
        # 添加网格
        if scalars is not None:
            plotter.add_mesh(
                mesh,
                scalars=scalars,
                cmap=cmap,
                opacity=opacity,
                show_edges=show_edges,
                show_scalar_bar=True
            )
        else:
            plotter.add_mesh(
                mesh,
                color='lightgray',
                opacity=opacity,
                show_edges=show_edges
            )
        
        # 设置视角
        plotter.view_isometric()
        plotter.add_axes()
        
        # 生成图像
        plotter.show(auto_close=False)
        image = plotter.screenshot()
        plotter.close()
        
        return image, "PyVista 3D"
        
    except Exception as e:
        error_msg = str(e)
        # 检查是否是图形库相关错误
        if any(keyword in error_msg.lower() for keyword in ['libxrender', 'libgl', 'display', 'x11', 'opengl']):
            raise Exception(f"图形库错误 (云端环境): {error_msg}")
        else:
            raise Exception(f"PyVista渲染失败: {error_msg}")

def create_matplotlib_3d_plot(mesh, scalars=None, cmap='viridis', opacity=0.8, show_edges=True, title="3D Visualization"):
    """使用matplotlib创建3D图像作为备选"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 获取网格点
        points = mesh.points
        
        if scalars is not None and len(scalars) == len(points):
            # 使用标量数据着色
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                               c=scalars, cmap=cmap, s=1, alpha=opacity)
            plt.colorbar(scatter, ax=ax, shrink=0.5)
        else:
            # 简单的点云显示
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                      c='blue', s=1, alpha=opacity)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(title)
        
        # 设置等比例
        max_range = np.array([points[:, 0].max()-points[:, 0].min(),
                             points[:, 1].max()-points[:, 1].min(),
                             points[:, 2].max()-points[:, 2].min()]).max() / 2.0
        mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # 保存为图像
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue(), "Matplotlib 3D"
        
    except Exception as e:
        raise Exception(f"Matplotlib 3D渲染失败: {str(e)}")

def create_2d_projection_plot(mesh, scalars=None, cmap='viridis', opacity=0.8, show_edges=True, title="2D Projection"):
    """创建2D投影图作为最后备选"""
    import matplotlib.pyplot as plt
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        points = mesh.points
        
        # XY投影
        if scalars is not None and len(scalars) == len(points):
            scatter1 = ax1.scatter(points[:, 0], points[:, 1], c=scalars, cmap=cmap, s=1, alpha=opacity)
            plt.colorbar(scatter1, ax=ax1)
        else:
            ax1.scatter(points[:, 0], points[:, 1], c='blue', s=1, alpha=opacity)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('XY投影')
        ax1.grid(True, alpha=0.3)
        
        # XZ投影
        if scalars is not None and len(scalars) == len(points):
            scatter2 = ax2.scatter(points[:, 0], points[:, 2], c=scalars, cmap=cmap, s=1, alpha=opacity)
        else:
            ax2.scatter(points[:, 0], points[:, 2], c='blue', s=1, alpha=opacity)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_title('XZ投影')
        ax2.grid(True, alpha=0.3)
        
        # YZ投影
        if scalars is not None and len(scalars) == len(points):
            scatter3 = ax3.scatter(points[:, 1], points[:, 2], c=scalars, cmap=cmap, s=1, alpha=opacity)
        else:
            ax3.scatter(points[:, 1], points[:, 2], c='blue', s=1, alpha=opacity)
        ax3.set_xlabel('Y')
        ax3.set_ylabel('Z')
        ax3.set_title('YZ投影')
        ax3.grid(True, alpha=0.3)
        
        # 统计信息
        ax4.axis('off')
        stats_text = f"""
        网格统计信息:
        • 点数: {mesh.n_points}
        • 单元数: {mesh.n_cells}
        • X范围: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]
        • Y范围: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]
        • Z范围: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]
        """
        if scalars is not None:
            stats_text += f"\n• 标量范围: [{scalars.min():.3f}, {scalars.max():.3f}]"
        
        ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        plt.suptitle(title)
        plt.tight_layout()
        
        # 保存为图像
        import io
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf.getvalue(), "2D Projections"
        
    except Exception as e:
        raise Exception(f"2D投影渲染失败: {str(e)}")

# 创建安全的交互式窗口函数
def create_safe_interactive_window(plotter_func, fallback_func=None):
    """创建安全的交互式窗口，处理中文显示和窗口管理问题"""
    try:
        # 重置PyVista状态
        pv.close_all()
        
        # 确保本地环境下的正确设置
        if not is_cloud_environment():
            pv.OFF_SCREEN = False
            pv.set_plot_theme("document")
        
        # 执行绘图函数
        result = plotter_func()
        
        # 强制关闭所有窗口以避免残留
        pv.close_all()
        
        return result, None
        
    except Exception as e:
        # 清理可能的残留窗口
        try:
            pv.close_all()
        except:
            pass
        
        error_msg = str(e)
        
        # 如果有备选方案，尝试执行
        if fallback_func:
            try:
                return fallback_func(), f"交互式窗口失败，使用备选方案: {error_msg}"
            except Exception as fallback_error:
                return None, f"交互式和备选方案都失败: {error_msg}, {str(fallback_error)}"
        
        return None, error_msg

# 初始化PyVista配置
configure_pyvista_for_cloud()

# 导入预测测试所需的库
try:
    from ezyrb import POD, RBF, Database, GPR, ANN, KNeighborsRegressor, RadiusNeighborsRegressor, PODAE, AE
    from ezyrb import ReducedOrderModel as ROM
    from sklearn.gaussian_process.kernels import RBF as RBFS, WhiteKernel, ConstantKernel, Matern
    from sklearn.preprocessing import StandardScaler
    EZYRB_AVAILABLE = True
except ImportError:
    EZYRB_AVAILABLE = False
    st.warning("⚠️ EZyRB库未安装，预测测试功能将不可用")

# 设置页面配置
st.set_page_config(
    page_title="模型降阶工具",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置全局样式
st.markdown("""
<style>
    /* 主容器样式 */
    .main {
        padding-top: 2rem;
    }
    
    /* 标题样式 */
    h1 {
        color: #1f77b4;
        border-bottom: 3px solid #1f77b4;
        padding-bottom: 10px;
        margin-bottom: 30px;
    }
    
    h2 {
        color: #2ca02c;
        margin-top: 20px;
        margin-bottom: 15px;
    }
    
    h3 {
        color: #ff7f0e;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    
    /* 信息框美化 */
    .stAlert {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    /* 按钮美化 */
    .stButton > button {
        border-radius: 20px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 10px rgba(0,0,0,0.2);
    }
    
    /* 侧边栏美化 */
    .css-1d391kg {
        background-color: #f0f2f6;
    }
    
    /* 分隔线样式 */
    hr {
        margin: 30px 0;
        border: none;
        border-top: 2px solid #e0e0e0;
    }
    
    /* 度量值卡片美化 */
    [data-testid="metric-container"] {
        background-color: #f8f9fa;
        border: 1px solid #e0e0e0;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* 文件上传区域美化 */
    [data-testid="stFileUploader"] {
        border: 2px dashed #cccccc;
        border-radius: 10px;
        padding: 20px;
        background-color: #fafafa;
    }
    
    /* 展开器美化 */
    .streamlit-expanderHeader {
        background-color: #f0f2f6;
        border-radius: 10px;
        font-weight: bold;
    }
    
    /* 选择框美化 */
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    /* 数据框美化 */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# 初始化session state
if 'snapshots_x' not in st.session_state:
    st.session_state.snapshots_x = None
if 'snapshots_y' not in st.session_state:
    st.session_state.snapshots_y = None
if 'snapshots_z' not in st.session_state:
    st.session_state.snapshots_z = None
if 'snapshots_stress' not in st.session_state:
    st.session_state.snapshots_stress = None
if 'param' not in st.session_state:
    st.session_state.param = None
if 'file_info' not in st.session_state:
    st.session_state.file_info = ""
if 'array_info' not in st.session_state:
    st.session_state.array_info = ""
if 'default_save_path' not in st.session_state:
    st.session_state.default_save_path = str(Path.cwd())
if 'generated_plots' not in st.session_state:
    st.session_state.generated_plots = []
if 'mesh_data' not in st.session_state:
    st.session_state.mesh_data = None
if 'mesh_info' not in st.session_state:
    st.session_state.mesh_info = ""

# 清除数组数据的函数
def clear_all_arrays():
    """清除所有已加载的数组数据"""
    st.session_state.snapshots_x = None
    st.session_state.snapshots_y = None
    st.session_state.snapshots_z = None
    st.session_state.snapshots_stress = None
    st.session_state.param = None
    st.session_state.file_info = ""
    st.session_state.array_info = ""
    st.session_state.mesh_data = None
    st.session_state.mesh_info = ""

# 更新数组信息的函数
def update_array_info():
    """更新数组信息显示"""
    info_parts = []
    if st.session_state.snapshots_x is not None:
        info_parts.append(f"X分量: {st.session_state.snapshots_x.shape}")
    if st.session_state.snapshots_y is not None:
        info_parts.append(f"Y分量: {st.session_state.snapshots_y.shape}")
    if st.session_state.snapshots_z is not None:
        info_parts.append(f"Z分量: {st.session_state.snapshots_z.shape}")
    if st.session_state.snapshots_stress is not None:
        info_parts.append(f"应力: {st.session_state.snapshots_stress.shape}")
    if st.session_state.param is not None:
        info_parts.append(f"参数: {st.session_state.param.shape}")
    
    if info_parts:
        st.session_state.array_info = "📋 已读取的数组:\n" + "\n".join([f"• {info}" for info in info_parts])
    else:
        st.session_state.array_info = ""

# 侧边栏页面选择
with st.sidebar:
    st.markdown("# 📊 模型降阶工具")
    st.markdown("---")
    
    # 添加简介
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <h4 style='margin: 0 0 10px 0; color: #1f77b4;'>🚀 功能简介</h4>
        <p style='margin: 0; font-size: 14px;'>
        • 支持大文件VTU/NPY导入<br>
        • 多种降阶方法对比<br>
        • 交互式3D可视化<br>
        • 预测精度分析<br>
        • 图表导出功能
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    page = st.selectbox(
        "🎯 选择功能页面",
        ["📥 数据导入与保存", "🔬 预测测试", "🔗 联合降阶模型测试", "🎨 三维可视化", "📈 图表输出"],
        help="选择您要使用的功能模块"
    )
    
    st.markdown("---")
    
    # 添加当前数据状态
    st.markdown("### 📊 数据状态")
    data_status = []
    if st.session_state.snapshots_x is not None:
        data_status.append(f"✅ X分量")
    if st.session_state.snapshots_y is not None:
        data_status.append(f"✅ Y分量")
    if st.session_state.snapshots_z is not None:
        data_status.append(f"✅ Z分量")
    if st.session_state.snapshots_stress is not None:
        data_status.append(f"✅ 应力")
    if st.session_state.param is not None:
        data_status.append(f"✅ 参数")
    
    if data_status:
        st.success("已加载: " + " | ".join(data_status))
    else:
        st.info("尚未加载任何数据")

# 页面1：数据导入与保存
if page == "📥 数据导入与保存":
    st.title("📥 数据导入与保存")
    
    # 添加页面描述
    st.markdown("""
    <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
        <h4 style='margin: 0 0 10px 0;'>📌 功能说明</h4>
        <p style='margin: 0;'>本页面支持从VTU文件提取位移数据，或直接导入NPY格式的数组文件。支持任意大小的文件上传，无需额外配置。</p>
    </div>
    """, unsafe_allow_html=True)

    # 创建三列布局：数据读取、数据管理、数据保存
    col1, col2, col3 = st.columns([1.2, 0.8, 1])

    with col1:
        st.markdown("## 🔍 数据读取")
        
        # 创建选项卡
        tab_vtu, tab_npy = st.tabs(["📄 VTU文件", "📦 NPY文件"])
        
        with tab_vtu:
            st.markdown("### 读取VTU文件")
            
            # 文件上传区域
            uploaded_vtu_file = st.file_uploader(
                "拖拽或点击上传VTU文件", 
                type=['vtu'],
                help="支持任意大小的VTU文件，包含位移数据"
            )
            
            if uploaded_vtu_file is not None:
                # 显示文件信息
                file_size = uploaded_vtu_file.size / (1024 * 1024)  # 转换为MB
                st.markdown(f"""
                <div style='background-color: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                    📁 文件名: {uploaded_vtu_file.name}<br>
                    📏 文件大小: {file_size:.2f} MB
                </div>
                """, unsafe_allow_html=True)
                
                # 保存上传的文件到临时目录
                with tempfile.NamedTemporaryFile(delete=False, suffix='.vtu') as temp_file:
                    temp_file.write(uploaded_vtu_file.getvalue())
                    temp_file_path = temp_file.name
                
                try:
                    # 列出可用的deltaT值
                    deltats = list_available_deltats(temp_file_path)
                    
                    if deltats:
                        st.success(f"✅ 成功读取VTU文件: {uploaded_vtu_file.name}")
                        st.info(f"📋 找到 {len(deltats)} 个deltaT值: {', '.join(deltats)}")
                        
                        # 让用户选择参数范围
                        st.markdown("### ⚙️ 参数设置")
                        
                        col_param1, col_param2, col_param3 = st.columns(3)
                        with col_param1:
                            param_start = st.number_input("起始值", value=-50, help="参数的起始值")
                        with col_param2:
                            param_end = st.number_input("结束值", value=90, help="参数的结束值")
                        with col_param3:
                            param_step = st.number_input("步长", value=20, min_value=1, help="参数的步长")
                        
                        if st.button("🚀 开始处理VTU数据", type="primary", use_container_width=True):
                            with st.spinner("正在处理数据..."):
                                try:
                                    snapshots_x = []
                                    snapshots_y = []
                                    snapshots_z = []
                                    snapshots_stress = []
                                    
                                    # 处理进度条
                                    progress_bar = st.progress(0)
                                    status_text = st.empty()
                                    
                                    for i, deltaT in enumerate(deltats):
                                        status_text.text(f"正在处理 deltaT={deltaT}...")
                                        
                                        x_data, y_data, z_data, stress, mesh, found_components = extract_displacement_components(
                                            temp_file_path,
                                            deltaT=deltaT,
                                            output_dir=None,
                                            visualize=False
                                        )
                                        
                                        if x_data is not None:
                                            snapshots_x.append(x_data)
                                        if y_data is not None:
                                            snapshots_y.append(y_data)
                                        if z_data is not None:
                                            snapshots_z.append(z_data)
                                        if stress is not None:
                                            snapshots_stress.append(stress)
                                        
                                        progress_bar.progress((i + 1) / len(deltats))
                                    
                                    # 转换为numpy数组
                                    if snapshots_x:
                                        st.session_state.snapshots_x = np.array(snapshots_x)
                                    if snapshots_y:
                                        st.session_state.snapshots_y = np.array(snapshots_y)
                                    if snapshots_z:
                                        st.session_state.snapshots_z = np.array(snapshots_z)
                                    if snapshots_stress:
                                        st.session_state.snapshots_stress = np.array(snapshots_stress)
                                    
                                    # 保存网格数据（使用最后一个mesh）
                                    if mesh is not None:
                                        st.session_state.mesh_data = mesh
                                        st.session_state.mesh_info = f"""
                                        📐 网格信息:
                                        • 点数: {mesh.n_points}
                                        • 单元数: {mesh.n_cells}
                                        • 边界: {mesh.bounds}
                                        """
                                    
                                    # 生成参数数组
                                    param_range = np.arange(param_start, param_end, param_step)
                                    st.session_state.param = param_range.reshape(-1, 1)
                                    
                                    # 更新文件信息
                                    st.session_state.file_info = f"""
                                    📁 文件名: {uploaded_vtu_file.name}
                                    📊 deltaT值数量: {len(deltats)}
                                    📈 参数范围: {param_start} 到 {param_end-param_step} (步长: {param_step})
                                    ✅ 处理完成时间: {np.datetime64('now')}
                                    """
                                    
                                    # 更新数组信息
                                    update_array_info()
                                    
                                    status_text.text("✅ 数据处理完成!")
                                    st.success("🎉 VTU数据处理完成!")
                                    
                                    # 显示处理结果摘要
                                    st.markdown(f"""
                                    <div style='background-color: #d4edda; padding: 15px; border-radius: 10px; margin-top: 20px;'>
                                        <h4 style='margin: 0 0 10px 0;'>✅ 处理成功</h4>
                                        <p style='margin: 5px 0;'>• 提取了 {len(deltats)} 个时间步的数据</p>
                                        <p style='margin: 5px 0;'>• 参数范围: {param_start} 到 {param_end-param_step}</p>
                                        <p style='margin: 5px 0;'>• 网格点数: {mesh.n_points if mesh else 'N/A'}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    
                                except Exception as e:
                                    st.error(f"❌ 处理VTU数据时出错: {str(e)}")
                    else:
                        st.error("❌ 未在文件中找到有效的deltaT数据")
                        
                except Exception as e:
                    st.error(f"❌ 读取VTU文件时出错: {str(e)}")
                finally:
                    # 清理临时文件
                    if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
        
        with tab_npy:
            st.markdown("### 读取NPY文件")
            
            uploaded_npy_files = st.file_uploader(
                "拖拽或点击上传NPY文件（支持多选）", 
                type=['npy'],
                accept_multiple_files=True,
                help="支持同时上传多个NPY文件，如snapshots_x.npy, snapshots_y.npy等"
            )
            
            if uploaded_npy_files:
                # 显示上传的文件列表
                st.markdown("#### 📋 已选择的文件:")
                file_info_html = "<div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px;'>"
                for file in uploaded_npy_files:
                    file_size = file.size / (1024 * 1024)
                    file_info_html += f"• {file.name} ({file_size:.2f} MB)<br>"
                file_info_html += "</div>"
                st.markdown(file_info_html, unsafe_allow_html=True)
                
                if st.button("📥 加载NPY数据", type="primary", use_container_width=True):
                    try:
                        loaded_files = []
                        with st.spinner("正在加载NPY文件..."):
                            for uploaded_file in uploaded_npy_files:
                                file_name = uploaded_file.name.lower()
                                data = np.load(uploaded_file)
                                
                                if 'snapshots_x' in file_name or 'x' in file_name:
                                    st.session_state.snapshots_x = data
                                    loaded_files.append(f"X分量: {uploaded_file.name} ({data.shape})")
                                elif 'snapshots_y' in file_name or 'y' in file_name:
                                    st.session_state.snapshots_y = data
                                    loaded_files.append(f"Y分量: {uploaded_file.name} ({data.shape})")
                                elif 'snapshots_z' in file_name or 'z' in file_name:
                                    st.session_state.snapshots_z = data
                                    loaded_files.append(f"Z分量: {uploaded_file.name} ({data.shape})")
                                elif 'stress' in file_name:
                                    st.session_state.snapshots_stress = data
                                    loaded_files.append(f"应力: {uploaded_file.name} ({data.shape})")
                                elif 'param' in file_name:
                                    st.session_state.param = data
                                    loaded_files.append(f"参数: {uploaded_file.name} ({data.shape})")
                                else:
                                    st.warning(f"⚠️ 未识别的文件: {uploaded_file.name}")
                        
                        # 更新文件信息
                        if loaded_files:
                            st.session_state.file_info = f"""
                            📁 数据来源: NPY文件
                            📊 加载的文件数量: {len(loaded_files)}
                            📋 文件详情:
                            {chr(10).join(['• ' + file for file in loaded_files])}
                            ✅ 加载完成时间: {np.datetime64('now')}
                            """
                        
                        # 更新数组信息
                        update_array_info()
                        
                        # 显示加载结果摘要
                        st.markdown(f"""
                        <div style='background-color: #d4edda; padding: 15px; border-radius: 10px; margin-top: 20px;'>
                            <h4 style='margin: 0 0 10px 0;'>✅ 加载成功</h4>
                            <p style='margin: 5px 0;'>成功加载 {len(loaded_files)} 个NPY文件</p>
                            {'<br>'.join([f"<p style='margin: 5px 0;'>• {file}</p>" for file in loaded_files])}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ 加载NPY文件时出错: {str(e)}")

    with col2:
        st.markdown("## 🗂️ 数据管理")
        
        # 使用卡片样式显示数据状态
        st.markdown("""
        <div style='background-color: #fff3cd; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h4 style='margin: 0 0 10px 0;'>📊 当前数据状态</h4>
        """, unsafe_allow_html=True)
        
        # 检查是否有数据
        has_data = any([
            st.session_state.snapshots_x is not None,
            st.session_state.snapshots_y is not None, 
            st.session_state.snapshots_z is not None,
            st.session_state.snapshots_stress is not None,
            st.session_state.param is not None
        ])
        
        if has_data:
            data_info = []
            if st.session_state.snapshots_x is not None:
                data_info.append(f"✅ X分量: {st.session_state.snapshots_x.shape}")
            if st.session_state.snapshots_y is not None:
                data_info.append(f"✅ Y分量: {st.session_state.snapshots_y.shape}")
            if st.session_state.snapshots_z is not None:
                data_info.append(f"✅ Z分量: {st.session_state.snapshots_z.shape}")
            if st.session_state.snapshots_stress is not None:
                data_info.append(f"✅ 应力: {st.session_state.snapshots_stress.shape}")
            if st.session_state.param is not None:
                data_info.append(f"✅ 参数: {st.session_state.param.shape}")
            
            for info in data_info:
                st.markdown(f"<p style='margin: 5px 0;'>{info}</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='margin: 0; color: #666;'>尚未加载任何数据</p>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        if has_data:
            st.markdown("### 🧹 清除数据")
            st.warning("⚠️ 此操作将清除所有已加载的数组数据")
            
            if st.button("🗑️ 清除所有数据", type="secondary", use_container_width=True):
                clear_all_arrays()
                st.success("✅ 所有数据已清除!")
                st.rerun()
        
        st.markdown("---")
        
        # 保存路径设置
        st.markdown("### 📁 保存路径设置")
        
        # 默认保存路径设置
        current_default = st.session_state.default_save_path
        new_default_path = st.text_input(
            "默认保存路径", 
            value=current_default,
            help="设置数据保存的默认路径"
        )
        
        col_path1, col_path2 = st.columns(2)
        with col_path1:
            if st.button("📂 选择当前目录", use_container_width=True):
                st.session_state.default_save_path = str(Path.cwd())
                st.rerun()
        
        with col_path2:
            if st.button("💾 更新默认路径", use_container_width=True):
                if Path(new_default_path).exists():
                    st.session_state.default_save_path = new_default_path
                    st.success("✅ 默认路径已更新!")
                else:
                    st.error("❌ 路径不存在!")
        
        st.info(f"📍 当前默认路径: {st.session_state.default_save_path}")

    with col3:
        st.markdown("## 💾 数据保存与信息")
        
        # 使用展开器组织内容
        with st.expander("💾 保存数据", expanded=True):
            if has_data:
                # 保存路径选择
                use_default = st.checkbox("使用默认保存路径", value=True)
                
                if use_default:
                    save_base_path = st.session_state.default_save_path
                    st.info(f"📍 保存路径: {save_base_path}")
                else:
                    save_base_path = st.text_input(
                        "自定义保存路径", 
                        value=st.session_state.default_save_path,
                        help="输入自定义的保存基础路径"
                    )
                
                save_folder_name = st.text_input(
                    "保存文件夹名称", 
                    value="displacement_data",
                    help="输入要保存数据的文件夹名称"
                )
                
                # 显示完整保存路径
                full_save_path = Path(save_base_path) / save_folder_name
                st.markdown(f"""
                <div style='background-color: #d1ecf1; padding: 10px; border-radius: 5px; margin: 10px 0;'>
                    📁 完整保存路径: {full_save_path}
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("💾 保存所有数据", type="primary", use_container_width=True):
                    try:
                        # 确保基础路径存在
                        base_path = Path(save_base_path)
                        if not base_path.exists():
                            st.error(f"❌ 基础路径不存在: {base_path}")
                        else:
                            # 创建保存目录
                            save_dir = base_path / save_folder_name
                            save_dir.mkdir(exist_ok=True)
                            
                            saved_files = []
                            
                            # 保存各个数组
                            if st.session_state.snapshots_x is not None:
                                file_path = save_dir / "snapshots_x.npy"
                                np.save(file_path, st.session_state.snapshots_x)
                                saved_files.append(f"snapshots_x.npy ({st.session_state.snapshots_x.shape})")
                            
                            if st.session_state.snapshots_y is not None:
                                file_path = save_dir / "snapshots_y.npy"
                                np.save(file_path, st.session_state.snapshots_y)
                                saved_files.append(f"snapshots_y.npy ({st.session_state.snapshots_y.shape})")
                            
                            if st.session_state.snapshots_z is not None:
                                file_path = save_dir / "snapshots_z.npy"
                                np.save(file_path, st.session_state.snapshots_z)
                                saved_files.append(f"snapshots_z.npy ({st.session_state.snapshots_z.shape})")
                            
                            if st.session_state.snapshots_stress is not None:
                                file_path = save_dir / "snapshots_stress.npy"
                                np.save(file_path, st.session_state.snapshots_stress)
                                saved_files.append(f"snapshots_stress.npy ({st.session_state.snapshots_stress.shape})")
                            
                            if st.session_state.param is not None:
                                file_path = save_dir / "param.npy"
                                np.save(file_path, st.session_state.param)
                                saved_files.append(f"param.npy ({st.session_state.param.shape})")
                            
                            st.success(f"✅ 数据已保存到文件夹: {save_dir.absolute()}")
                            st.info("📁 保存的文件:\n" + "\n".join([f"• {file}" for file in saved_files]))
                            
                            # 显示保存成功提示
                            st.markdown(f"""
                            <div style='background-color: #d4edda; padding: 15px; border-radius: 10px; margin-top: 20px;'>
                                <h4 style='margin: 0 0 10px 0;'>✅ 保存成功</h4>
                                <p style='margin: 5px 0;'>数据已保存到: {save_dir.absolute()}</p>
                                <p style='margin: 5px 0;'>共保存 {len(saved_files)} 个文件</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f"❌ 保存数据时出错: {str(e)}")
            else:
                st.info("ℹ️ 没有可保存的数据，请先读取VTU或NPY文件")
        
        # 信息显示部分
        with st.expander("📊 数据信息", expanded=True):
            if st.session_state.file_info:
                st.markdown("#### 📁 文件信息")
                st.text_area("", st.session_state.file_info, height=120, disabled=True, label_visibility="collapsed")
            else:
                st.info("ℹ️ 尚未读取任何文件")
            
            if st.session_state.array_info:
                st.markdown("#### 📋 数组信息")
                # 添加刷新按钮
                if st.button("🔄 刷新状态", key="refresh_array_info"):
                    update_array_info()
                    st.success("✅ 状态已刷新!")
                st.text_area("", st.session_state.array_info, height=120, disabled=True, label_visibility="collapsed")
            else:
                st.info("ℹ️ 尚未加载任何数组")
        
        # 数据预览
        if has_data:
            with st.expander("🔍 数据预览", expanded=False):
                preview_option = st.selectbox(
                    "选择要预览的数据",
                    ["无", "X分量", "Y分量", "Z分量", "应力", "参数"]
                )
                
                if preview_option != "无":
                    data_map = {
                        "X分量": st.session_state.snapshots_x,
                        "Y分量": st.session_state.snapshots_y,
                        "Z分量": st.session_state.snapshots_z,
                        "应力": st.session_state.snapshots_stress,
                        "参数": st.session_state.param
                    }
                    
                    selected_data = data_map[preview_option]
                    if selected_data is not None:
                        # 使用度量值卡片显示统计信息
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            st.metric("数据形状", str(selected_data.shape))
                        with col_stat2:
                            st.metric("数据类型", str(selected_data.dtype))
                        with col_stat3:
                            st.metric("数据范围", f"[{selected_data.min():.3f}, {selected_data.max():.3f}]")
                        
                        # 显示前几行数据
                        st.markdown("#### 数据样本")
                        if len(selected_data.shape) == 1:
                            st.write("前10个值:")
                            st.code(str(selected_data[:10]))
                        else:
                            st.write("前5行数据:")
                            st.dataframe(selected_data[:5], use_container_width=True)
                    else:
                        st.warning(f"⚠️ {preview_option}数据未加载")

# 页面2：预测测试
elif page == "🔬 预测测试":
    st.title("🔬 预测测试")
    
    # 添加页面描述
    st.markdown("""
    <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
        <h4 style='margin: 0 0 10px 0;'>📌 功能说明</h4>
        <p style='margin: 0;'>使用降阶模型进行预测测试，支持单点验证、多点验证和K折交叉验证，评估模型预测精度。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 显示数据概览卡片
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <h4 style='margin: 0 0 10px 0;'>📊 数据概览</h4>
    """, unsafe_allow_html=True)
    
    data_overview = []
    if st.session_state.snapshots_x is not None:
        data_overview.append(f"X分量: {st.session_state.snapshots_x.shape}")
    if st.session_state.snapshots_y is not None:
        data_overview.append(f"Y分量: {st.session_state.snapshots_y.shape}")
    if st.session_state.snapshots_z is not None:
        data_overview.append(f"Z分量: {st.session_state.snapshots_z.shape}")
    if st.session_state.snapshots_stress is not None:
        data_overview.append(f"应力: {st.session_state.snapshots_stress.shape}")
    if st.session_state.param is not None:
        data_overview.append(f"参数: {st.session_state.param.shape}")
    
    if data_overview:
        st.markdown("<p style='margin: 0;'>✅ 已加载数据: " + " | ".join(data_overview) + "</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='margin: 0; color: #666;'>ℹ️ 尚未加载任何数据</p>", unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if not EZYRB_AVAILABLE:
        st.error("❌ EZyRB库未安装，无法使用预测测试功能")
        st.info("请安装EZyRB库：pip install ezyrb")
        st.stop()
    
    # 检查是否有可用数据
    available_data = {}
    if st.session_state.snapshots_x is not None:
        available_data["X分量"] = st.session_state.snapshots_x
    if st.session_state.snapshots_y is not None:
        available_data["Y分量"] = st.session_state.snapshots_y
    if st.session_state.snapshots_z is not None:
        available_data["Z分量"] = st.session_state.snapshots_z
    if st.session_state.snapshots_stress is not None:
        available_data["应力"] = st.session_state.snapshots_stress
    
    # 显示当前数据状态（调试信息）
    with st.expander("🔍 当前数据状态", expanded=False):
        st.write("**Session State 数据检查**:")
        st.write(f"- snapshots_x: {st.session_state.snapshots_x is not None} {st.session_state.snapshots_x.shape if st.session_state.snapshots_x is not None else 'None'}")
        st.write(f"- snapshots_y: {st.session_state.snapshots_y is not None} {st.session_state.snapshots_y.shape if st.session_state.snapshots_y is not None else 'None'}")
        st.write(f"- snapshots_z: {st.session_state.snapshots_z is not None} {st.session_state.snapshots_z.shape if st.session_state.snapshots_z is not None else 'None'}")
        st.write(f"- snapshots_stress: {st.session_state.snapshots_stress is not None} {st.session_state.snapshots_stress.shape if st.session_state.snapshots_stress is not None else 'None'}")
        st.write(f"- param: {st.session_state.param is not None} {st.session_state.param.shape if st.session_state.param is not None else 'None'}")
        st.write(f"**可用数据类型**: {list(available_data.keys())}")
    
    if not available_data:
        st.error("❌ 没有找到任何快照数据（X分量、Y分量、Z分量或应力数据）")
        st.info("💡 请在'数据导入与保存'页面加载VTU文件或NPY文件")
        st.stop()
    
    if st.session_state.param is None:
        st.error("❌ 没有找到参数数据")
        st.info("💡 请确保已加载参数数据（param.npy文件或通过VTU文件生成）")
        st.stop()
    
    # 创建两个选项卡
    tab1, tab2 = st.tabs(["🎯 参数预测测试", "🔄 K折交叉验证"])
    
    with tab1:
        st.header("🎯 参数预测测试")
        
        # 选择验证模式
        validation_mode = st.radio(
            "选择验证模式",
            ["🎯 单点验证", "📊 多点验证"],
            horizontal=True,
            help="单点验证：逐个验证每个选择的点；多点验证：同时验证多个点并在同一图表中显示"
        )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 数据选择")
            
            # 选择快照数据
            selected_snapshot_type = st.selectbox(
                "选择快照数据类型",
                list(available_data.keys())
            )
            selected_snapshots = available_data[selected_snapshot_type]
            
            st.info(f"已选择: {selected_snapshot_type} - 形状: {selected_snapshots.shape}")
            
            # 参数数据
            max_points = len(st.session_state.param)
            st.info(f"参数数据形状: {st.session_state.param.shape} (共{max_points}个参数点)")
            
            st.subheader("⚙️ 验证参数设置")
            
            # 选择验证数据点数量
            n_validation = st.slider(
                "验证数据点数量 (n)",
                min_value=1,
                max_value=max_points-1,
                value=min(3, max_points-1),
                help=f"从{max_points}个参数点中选择n个作为验证数据，其余{max_points}个点构成训练集"
            )
            
            # 显示训练集大小
            st.info(f"训练集大小: {max_points - n_validation} 个点")
            
            # 选择验证数据的索引
            validation_indices = st.multiselect(
                "选择验证数据索引",
                options=list(range(max_points)),
                default=list(range(min(n_validation, 3))),
                max_selections=n_validation,
                help="选择哪些参数点用作验证数据"
            )
            
            if len(validation_indices) != n_validation:
                st.warning(f"⚠️ 请选择 {n_validation} 个验证数据索引")
            
            # 显示验证模式说明
            if validation_mode == "🎯 单点验证":
                st.info("💡 单点验证模式：每个验证点单独训练模型并生成独立的图表")
            else:
                st.info("💡 多点验证模式：所有验证点结果将显示在同一张图表中进行对比")
        
        with col2:
            st.subheader("🔧 模型配置")
            
            # 降阶方法选择
            reduction_method = st.selectbox(
                "选择降阶方法",
                ["POD", "PODAE", "AE"],
                help="选择降阶方法"
            )
            
            # 近似方法选择
            approximation_method = st.selectbox(
                "选择近似方法",
                ["RBF", "GPR", "ANN", "KNeighborsRegressor"],
                help="选择近似方法"
            )
            
            # RBF参数设置
            if approximation_method == "RBF":
                rbf_kernel = st.selectbox(
                    "RBF核函数",
                    ["multiquadric", "inverse", "gaussian", "linear", "cubic", "quintic", "thin_plate"]
                )
                rbf_epsilon = st.number_input("RBF epsilon", value=0.02, min_value=0.001, max_value=1.0, step=0.001)
            
            # GPR参数设置
            if approximation_method == "GPR":
                st.subheader("GPR参数设置")
                
                # 核函数选择
                gpr_kernel_type = st.selectbox(
                    "GPR核函数类型",
                    ["RBF", "Matern", "RationalQuadratic", "ExpSineSquared", "DotProduct", "WhiteKernel+RBF"],
                    help="选择高斯过程的核函数类型"
                )
                
                # Matern核函数的nu参数
                if gpr_kernel_type == "Matern":
                    matern_nu = st.selectbox(
                        "Matern nu参数",
                        [0.5, 1.5, 2.5, float('inf')],
                        index=1,
                        help="nu=0.5对应指数核，nu=1.5对应一阶可导，nu=2.5对应二阶可导，nu=inf对应RBF核"
                    )
                
                # 长度尺度参数
                gpr_length_scale = st.number_input(
                    "长度尺度 (length_scale)",
                    value=1.0,
                    min_value=0.01,
                    max_value=100.0,
                    step=0.1,
                    help="控制核函数的平滑程度"
                )
                
                # 长度尺度边界
                col_ls1, col_ls2 = st.columns(2)
                with col_ls1:
                    length_scale_bounds_min = st.number_input(
                        "长度尺度下界",
                        value=1e-5,
                        min_value=1e-10,
                        max_value=1.0,
                        format="%.2e",
                        help="优化时长度尺度的最小值"
                    )
                with col_ls2:
                    length_scale_bounds_max = st.number_input(
                        "长度尺度上界",
                        value=1e5,
                        min_value=1.0,
                        max_value=1e10,
                        format="%.2e",
                        help="优化时长度尺度的最大值"
                    )
                
                # 优化重启次数
                gpr_n_restarts = st.number_input(
                    "优化重启次数",
                    value=10,
                    min_value=0,
                    step=1,
                    help="核函数超参数优化的重启次数，越大越可能找到全局最优（注意：次数越多计算时间越长）"
                )
                
                # 是否归一化
                gpr_normalize = st.checkbox(
                    "归一化数据",
                    value=False,
                    help="是否对输入数据进行归一化处理"
                )
            
            st.subheader("🚀 预测执行")
            
            if len(validation_indices) == n_validation:
                button_text = "🎯 开始单点验证" if validation_mode == "🎯 单点验证" else "📊 开始多点验证"
                if st.button(button_text, type="primary"):
                    with st.spinner("正在进行预测测试..."):
                        try:
                            # 准备数据
                            param_data = st.session_state.param
                            snapshot_data = selected_snapshots
                            
                            if validation_mode == "🎯 单点验证":
                                # 单点验证模式：每个点单独训练模型
                                results = []
                                
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                
                                for i, val_idx in enumerate(validation_indices):
                                    status_text.text(f"正在验证第 {i+1}/{len(validation_indices)} 个点 (索引: {val_idx})...")
                                    
                                    # 获取验证数据
                                    validation_param = param_data[val_idx]
                                    validation_snapshot = snapshot_data[val_idx]
                                    validation_mean = np.mean(validation_snapshot)
                                    
                                    # 构建训练数据集（排除当前验证点）
                                    training_indices = list(range(len(param_data)))
                                    training_indices.remove(val_idx)
                                    training_params = param_data[training_indices]
                                    training_snapshots = snapshot_data[training_indices]
                                    
                                    # 构建数据库
                                    db = Database(training_params, training_snapshots)
                                    
                                    # 选择降阶方法
                                    if reduction_method == "POD":
                                        reducer = POD()
                                    elif reduction_method == "PODAE":
                                        reducer = PODAE()
                                    elif reduction_method == "AE":
                                        reducer = AE()
                                    
                                    # 选择近似方法
                                    if approximation_method == "RBF":
                                        approximator = RBF(kernel=rbf_kernel, epsilon=rbf_epsilon)
                                    elif approximation_method == "GPR":
                                        # 构建GPR核函数
                                        from sklearn.gaussian_process.kernels import RBF as RBFGP, Matern, RationalQuadratic, ExpSineSquared, DotProduct, WhiteKernel, ConstantKernel
                                        
                                        if gpr_kernel_type == "RBF":
                                            kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * RBFGP(
                                                length_scale=gpr_length_scale, 
                                                length_scale_bounds=(length_scale_bounds_min, length_scale_bounds_max)
                                            )
                                        elif gpr_kernel_type == "Matern":
                                            kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * Matern(
                                                length_scale=gpr_length_scale,
                                                length_scale_bounds=(length_scale_bounds_min, length_scale_bounds_max),
                                                nu=matern_nu
                                            )
                                        elif gpr_kernel_type == "RationalQuadratic":
                                            kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * RationalQuadratic(
                                                length_scale=gpr_length_scale,
                                                length_scale_bounds=(length_scale_bounds_min, length_scale_bounds_max)
                                            )
                                        elif gpr_kernel_type == "ExpSineSquared":
                                            kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * ExpSineSquared(
                                                length_scale=gpr_length_scale,
                                                length_scale_bounds=(length_scale_bounds_min, length_scale_bounds_max)
                                            )
                                        elif gpr_kernel_type == "DotProduct":
                                            kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * DotProduct()
                                        elif gpr_kernel_type == "WhiteKernel+RBF":
                                            kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * RBFGP(
                                                length_scale=gpr_length_scale,
                                                length_scale_bounds=(length_scale_bounds_min, length_scale_bounds_max)
                                            ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
                                        
                                        approximator = GPR(
                                            kern=kernel,
                                            normalizer=gpr_normalize,
                                            optimization_restart=gpr_n_restarts
                                        )
                                    elif approximation_method == "ANN":
                                        approximator = ANN()
                                    elif approximation_method == "KNeighborsRegressor":
                                        approximator = KNeighborsRegressor()
                                    
                                    # 构建ROM模型
                                    rom = ROM(db, reducer, approximator)
                                    rom.fit()
                                    
                                    # 预测
                                    result_db = rom.predict([validation_param])
                                    predicted_snapshot = result_db.snapshots_matrix.flatten()
                                    predicted_mean = np.mean(predicted_snapshot)
                                    
                                    # 计算误差
                                    error = np.abs(validation_snapshot - predicted_snapshot)
                                    max_error_idx = np.argmax(error)
                                    max_error = error[max_error_idx]
                                    relative_error = max_error / np.abs(validation_snapshot[max_error_idx]) * 100
                                    
                                    # 随机选择一个点
                                    random_idx = np.random.randint(0, len(validation_snapshot))
                                    
                                    results.append({
                                        'validation_idx': val_idx,
                                        'validation_param': validation_param,
                                        'validation_snapshot': validation_snapshot,
                                        'predicted_snapshot': predicted_snapshot,
                                        'validation_mean': validation_mean,
                                        'predicted_mean': predicted_mean,
                                        'training_params': training_params,
                                        'training_snapshots': training_snapshots,
                                        'max_error_idx': max_error_idx,
                                        'max_error': max_error,
                                        'relative_error': relative_error,
                                        'random_idx': random_idx
                                    })
                                    
                                    progress_bar.progress((i + 1) / len(validation_indices))
                                
                                status_text.text("✅ 单点验证完成!")
                                
                            else:
                                # 多点验证模式：一次性训练，验证多个点
                                st.info("🔄 多点验证模式：使用所有非验证点训练一个模型，然后预测所有验证点")
                                
                                # 构建训练数据集（排除所有验证点）
                                training_indices = [i for i in range(len(param_data)) if i not in validation_indices]
                                training_params = param_data[training_indices]
                                training_snapshots = snapshot_data[training_indices]
                                
                                st.info(f"训练数据: {len(training_indices)} 个点，验证数据: {len(validation_indices)} 个点")
                                
                                # 构建数据库
                                db = Database(training_params, training_snapshots)
                                
                                # 选择降阶方法
                                if reduction_method == "POD":
                                    reducer = POD()
                                elif reduction_method == "PODAE":
                                    reducer = PODAE()
                                elif reduction_method == "AE":
                                    reducer = AE()
                                
                                # 选择近似方法
                                if approximation_method == "RBF":
                                    approximator = RBF(kernel=rbf_kernel, epsilon=rbf_epsilon)
                                elif approximation_method == "GPR":
                                    # 构建GPR核函数
                                    from sklearn.gaussian_process.kernels import RBF as RBFGP, Matern, RationalQuadratic, ExpSineSquared, DotProduct, WhiteKernel, ConstantKernel
                                    
                                    if gpr_kernel_type == "RBF":
                                        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * RBFGP(
                                            length_scale=gpr_length_scale, 
                                            length_scale_bounds=(length_scale_bounds_min, length_scale_bounds_max)
                                        )
                                    elif gpr_kernel_type == "Matern":
                                        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * Matern(
                                            length_scale=gpr_length_scale,
                                            length_scale_bounds=(length_scale_bounds_min, length_scale_bounds_max),
                                            nu=matern_nu
                                        )
                                    elif gpr_kernel_type == "RationalQuadratic":
                                        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * RationalQuadratic(
                                            length_scale=gpr_length_scale,
                                            length_scale_bounds=(length_scale_bounds_min, length_scale_bounds_max)
                                        )
                                    elif gpr_kernel_type == "ExpSineSquared":
                                        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * ExpSineSquared(
                                            length_scale=gpr_length_scale,
                                            length_scale_bounds=(length_scale_bounds_min, length_scale_bounds_max)
                                        )
                                    elif gpr_kernel_type == "DotProduct":
                                        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * DotProduct()
                                    elif gpr_kernel_type == "WhiteKernel+RBF":
                                        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * RBFGP(
                                            length_scale=gpr_length_scale,
                                            length_scale_bounds=(length_scale_bounds_min, length_scale_bounds_max)
                                        ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
                                    
                                    approximator = GPR(
                                        kern=kernel,
                                        normalizer=gpr_normalize,
                                        optimization_restart=gpr_n_restarts
                                    )
                                elif approximation_method == "ANN":
                                    approximator = ANN()
                                elif approximation_method == "KNeighborsRegressor":
                                    approximator = KNeighborsRegressor()
                                
                                # 构建ROM模型
                                rom = ROM(db, reducer, approximator)
                                rom.fit()
                                
                                # 预测所有验证点
                                validation_params = param_data[validation_indices]
                                result_db = rom.predict(validation_params)
                                predicted_snapshots = result_db.snapshots_matrix
                                
                                # 处理结果
                                results = []
                                for i, val_idx in enumerate(validation_indices):
                                    validation_param = param_data[val_idx]
                                    validation_snapshot = snapshot_data[val_idx]
                                    predicted_snapshot = predicted_snapshots[i]
                                    
                                    validation_mean = np.mean(validation_snapshot)
                                    predicted_mean = np.mean(predicted_snapshot)
                                    
                                    # 计算误差
                                    error = np.abs(validation_snapshot - predicted_snapshot)
                                    max_error_idx = np.argmax(error)
                                    max_error = error[max_error_idx]
                                    relative_error = max_error / np.abs(validation_snapshot[max_error_idx]) * 100
                                    
                                    # 随机选择一个点
                                    random_idx = np.random.randint(0, len(validation_snapshot))
                                    
                                    results.append({
                                        'validation_idx': val_idx,
                                        'validation_param': validation_param,
                                        'validation_snapshot': validation_snapshot,
                                        'predicted_snapshot': predicted_snapshot,
                                        'validation_mean': validation_mean,
                                        'predicted_mean': predicted_mean,
                                        'training_params': training_params,
                                        'training_snapshots': training_snapshots,
                                        'max_error_idx': max_error_idx,
                                        'max_error': max_error,
                                        'relative_error': relative_error,
                                        'random_idx': random_idx
                                    })
                            
                            # 保存结果到session state
                            st.session_state.prediction_results = results
                            st.session_state.prediction_config = {
                                'snapshot_type': selected_snapshot_type,
                                'reduction_method': reduction_method,
                                'approximation_method': approximation_method,
                                'rbf_kernel': rbf_kernel if approximation_method == "RBF" else None,
                                'rbf_epsilon': rbf_epsilon if approximation_method == "RBF" else None,
                                'validation_mode': validation_mode,
                                'n_validation': n_validation,
                                'validation_indices': validation_indices
                            }
                            
                            st.success("✅ 预测测试完成!")
                            
                        except Exception as e:
                            st.error(f"❌ 预测测试失败: {str(e)}")
        
        # 显示预测结果
        if hasattr(st.session_state, 'prediction_results'):
            st.markdown("---")
            st.subheader("📊 预测结果")
            
            results = st.session_state.prediction_results
            config = st.session_state.prediction_config
            validation_mode = config.get('validation_mode', '🎯 单点验证')
            
            # 结果概览
            st.info(f"📋 测试配置: {config['snapshot_type']} | {config['reduction_method']} + {config['approximation_method']} | {validation_mode}")
            
            if validation_mode == "🎯 单点验证":
                # 单点验证模式：每个验证点单独显示图表
                st.write("**单点验证结果 - 每个验证点的独立分析**")
                
                for i, result in enumerate(results):
                    with st.expander(f"验证点 {result['validation_idx'] + 1} (参数值: {result['validation_param'].flatten()[0]:.2f})", expanded=i==0):
                        col_plot1, col_plot2 = st.columns(2)
                        
                        with col_plot1:
                            # 平均值对比图
                            fig1, ax1 = plt.subplots(figsize=(10, 6))
                            training_means = np.mean(result['training_snapshots'], axis=1)
                            ax1.scatter(result['training_params'], training_means, c='blue', alpha=0.6, label='Training data')
                            ax1.scatter(result['validation_param'], result['predicted_mean'], c='red', alpha=0.8, label='Prediction')
                            ax1.scatter(result['validation_param'], result['validation_mean'], c='green', alpha=0.8, label='Validation')
                            ax1.set_xlabel('Parameter')
                            ax1.set_ylabel('Mean Value')
                            ax1.set_title(f'Mean Value Comparison - Point {result["validation_idx"] + 1}')
                            ax1.grid(True, linestyle='--', alpha=0.7)
                            ax1.legend()
                            st.pyplot(fig1)
                        
                        with col_plot2:
                            # 最大误差点对比图
                            fig2, ax2 = plt.subplots(figsize=(10, 6))
                            training_means = np.mean(result['training_snapshots'], axis=1)
                            ax2.scatter(result['training_params'], training_means, c='blue', alpha=0.6, label='Training data')
                            ax2.scatter(result['validation_param'], result['predicted_snapshot'][result['max_error_idx']], 
                                       c='red', alpha=0.8, label=f'Max Error Prediction (idx:{result["max_error_idx"]})')
                            ax2.scatter(result['validation_param'], result['validation_snapshot'][result['max_error_idx']], 
                                       c='green', alpha=0.8, label=f'Max Error Validation')
                            ax2.set_xlabel('Parameter')
                            ax2.set_ylabel('Value')
                            ax2.set_title(f'Max Error Point - Relative Error: {result["relative_error"]:.2f}%')
                            ax2.grid(True, linestyle='--', alpha=0.7)
                            ax2.legend()
                            st.pyplot(fig2)
                        
                        # 点对点对比图
                        fig3, ax3 = plt.subplots(figsize=(12, 6))
                        x_indices = np.arange(len(result['validation_snapshot']))
                        ax3.scatter(x_indices, result['validation_snapshot'], c='green', alpha=0.6, label='Real Data')
                        ax3.scatter(x_indices, result['predicted_snapshot'], c='red', alpha=0.6, label='Predicted Data')
                        ax3.set_xlabel('Data Point Index')
                        ax3.set_ylabel('Value')
                        ax3.set_title(f'Point-by-Point Comparison - Validation Point {result["validation_idx"] + 1}')
                        ax3.grid(True, linestyle='--', alpha=0.7)
                        ax3.legend()
                        st.pyplot(fig3)
                        
                        # 显示误差统计
                        col_metric1, col_metric2, col_metric3 = st.columns(3)
                        with col_metric1:
                            st.metric("相对误差", f"{result['relative_error']:.2f}%")
                        with col_metric2:
                            st.metric("平均绝对误差", f"{np.mean(np.abs(result['validation_snapshot'] - result['predicted_snapshot'])):.2e}")
                        with col_metric3:
                            st.metric("最大绝对误差", f"{result['max_error']:.2e}")
                        
                        # 保存图表到session state
                        plot_info = {
                            'type': 'parameter_prediction_single',
                            'title': f'Single Point Prediction - Point {result["validation_idx"] + 1}',
                            'figures': [fig1, fig2, fig3],
                            'config': config,
                            'validation_idx': result['validation_idx']
                        }
                        if 'generated_plots' not in st.session_state:
                            st.session_state.generated_plots = []
                        st.session_state.generated_plots.append(plot_info)
            
            else:
                # 多点验证模式：所有验证点在同一图表中显示
                st.write("**多点验证结果 - 所有验证点的综合对比**")
                
                # 准备数据
                training_params = results[0]['training_params']  # 所有验证点使用相同的训练数据
                training_snapshots = results[0]['training_snapshots']
                training_means = np.mean(training_snapshots, axis=1)
                
                validation_params = [result['validation_param'].flatten()[0] for result in results]
                validation_means = [result['validation_mean'] for result in results]
                predicted_means = [result['predicted_mean'] for result in results]
                validation_indices = [result['validation_idx'] for result in results]
                
                col_plot1, col_plot2 = st.columns(2)
                
                with col_plot1:
                    # 综合平均值对比图
                    fig1, ax1 = plt.subplots(figsize=(12, 8))
                    ax1.scatter(training_params, training_means, c='blue', alpha=0.6, s=50, label='Training data')
                    
                    # 为每个验证点使用不同颜色
                    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
                    for i, (param, pred_mean, val_mean, val_idx) in enumerate(zip(validation_params, predicted_means, validation_means, validation_indices)):
                        ax1.scatter(param, pred_mean, c=[colors[i]], alpha=0.8, s=100, marker='^', label=f'Prediction Point {val_idx+1}')
                        ax1.scatter(param, val_mean, c=[colors[i]], alpha=0.8, s=100, marker='o', label=f'Validation Point {val_idx+1}')
                    
                    ax1.set_xlabel('Parameter')
                    ax1.set_ylabel('Mean Value')
                    ax1.set_title(f'Multi-Point Validation - Mean Value Comparison')
                    ax1.grid(True, linestyle='--', alpha=0.7)
                    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    plt.tight_layout()
                    st.pyplot(fig1)
                
                with col_plot2:
                    # 误差对比柱状图
                    fig2, ax2 = plt.subplots(figsize=(12, 8))
                    relative_errors = [result['relative_error'] for result in results]
                    point_labels = [f'Point {result["validation_idx"]+1}' for result in results]
                    
                    bars = ax2.bar(point_labels, relative_errors, color=colors, alpha=0.7)
                    ax2.set_xlabel('Validation Points')
                    ax2.set_ylabel('Relative Error (%)')
                    ax2.set_title('Multi-Point Validation - Relative Error Comparison')
                    ax2.grid(True, linestyle='--', alpha=0.3)
                    
                    # 在柱子上标注数值
                    for bar, error in zip(bars, relative_errors):
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{error:.2f}%', ha='center', va='bottom')
                    
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig2)
                
                # 综合点对点对比图（选择一个代表性的验证点）
                if len(results) > 0:
                    # 选择误差最大的点作为代表
                    max_error_result = max(results, key=lambda x: x['relative_error'])
                    
                    fig3, ax3 = plt.subplots(figsize=(12, 6))
                    x_indices = np.arange(len(max_error_result['validation_snapshot']))
                    ax3.scatter(x_indices, max_error_result['validation_snapshot'], c='green', alpha=0.6, label='Real Data')
                    ax3.scatter(x_indices, max_error_result['predicted_snapshot'], c='red', alpha=0.6, label='Predicted Data')
                    ax3.set_xlabel('Data Point Index')
                    ax3.set_ylabel('Value')
                    ax3.set_title(f'Point-by-Point Comparison - Worst Case (Point {max_error_result["validation_idx"] + 1})')
                    ax3.grid(True, linestyle='--', alpha=0.7)
                    ax3.legend()
                    st.pyplot(fig3)
                
                # 显示综合统计
                st.subheader("📈 综合统计")
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                all_relative_errors = [result['relative_error'] for result in results]
                all_abs_errors = [np.mean(np.abs(result['validation_snapshot'] - result['predicted_snapshot'])) for result in results]
                
                with col_stat1:
                    st.metric("平均相对误差", f"{np.mean(all_relative_errors):.2f}%")
                with col_stat2:
                    st.metric("最大相对误差", f"{np.max(all_relative_errors):.2f}%")
                with col_stat3:
                    st.metric("最小相对误差", f"{np.min(all_relative_errors):.2f}%")
                with col_stat4:
                    st.metric("相对误差标准差", f"{np.std(all_relative_errors):.2f}%")
                
                # 保存图表到session state
                plot_info = {
                    'type': 'parameter_prediction_multi',
                    'title': f'Multi-Point Prediction - {len(results)} Points',
                    'figures': [fig1, fig2, fig3],
                    'config': config,
                    'validation_indices': validation_indices,
                    'statistics': {
                        'mean_relative_error': np.mean(all_relative_errors),
                        'max_relative_error': np.max(all_relative_errors),
                        'min_relative_error': np.min(all_relative_errors),
                        'std_relative_error': np.std(all_relative_errors)
                    }
                }
                if 'generated_plots' not in st.session_state:
                    st.session_state.generated_plots = []
                st.session_state.generated_plots.append(plot_info)
    
    with tab2:
        st.header("🔄 K折交叉验证")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 数据选择")
            
            # 选择快照数据
            selected_snapshot_type_kfold = st.selectbox(
                "选择快照数据类型",
                list(available_data.keys()),
                key="kfold_snapshot_type"
            )
            selected_snapshots_kfold = available_data[selected_snapshot_type_kfold]
            
            st.info(f"已选择: {selected_snapshot_type_kfold} - 形状: {selected_snapshots_kfold.shape}")
            st.info(f"参数数据形状: {st.session_state.param.shape}")
            
            st.subheader("⚙️ K折设置")
            
            # K值选择
            max_k = len(st.session_state.param)
            k_value = st.slider(
                "K值 (折数)",
                min_value=2,
                max_value=max_k,
                value=min(5, max_k),
                help=f"K折交叉验证的折数（最大可设为参数点数：{max_k}）"
            )
        
        with col2:
            st.subheader("🔧 模型配置")
            
            # 降阶方法选择
            reduction_method_kfold = st.selectbox(
                "选择降阶方法",
                ["POD", "PODAE", "AE"],
                key="kfold_reduction_method"
            )
            
            # 近似方法选择
            approximation_method_kfold = st.selectbox(
                "选择近似方法",
                ["RBF", "GPR", "ANN", "KNeighborsRegressor"],
                key="kfold_approximation_method"
            )
            
            # RBF参数设置
            if approximation_method_kfold == "RBF":
                rbf_kernel_kfold = st.selectbox(
                    "RBF核函数",
                    ["multiquadric", "inverse", "gaussian", "linear", "cubic", "quintic", "thin_plate"],
                    key="kfold_rbf_kernel"
                )
                rbf_epsilon_kfold = st.number_input(
                    "RBF epsilon", 
                    value=0.02, 
                    min_value=0.001, 
                    max_value=1.0, 
                    step=0.001,
                    key="kfold_rbf_epsilon"
                )
            
            # GPR参数设置
            if approximation_method_kfold == "GPR":
                st.subheader("GPR参数设置")
                
                # 核函数选择
                gpr_kernel_type_kfold = st.selectbox(
                    "GPR核函数类型",
                    ["RBF", "Matern", "RationalQuadratic", "ExpSineSquared", "DotProduct", "WhiteKernel+RBF"],
                    key="kfold_gpr_kernel_type",
                    help="选择高斯过程的核函数类型"
                )
                
                # Matern核函数的nu参数
                if gpr_kernel_type_kfold == "Matern":
                    matern_nu_kfold = st.selectbox(
                        "Matern nu参数",
                        [0.5, 1.5, 2.5, float('inf')],
                        index=1,
                        key="kfold_matern_nu",
                        help="nu=0.5对应指数核，nu=1.5对应一阶可导，nu=2.5对应二阶可导，nu=inf对应RBF核"
                    )
                
                # 长度尺度参数
                gpr_length_scale_kfold = st.number_input(
                    "长度尺度 (length_scale)",
                    value=1.0,
                    min_value=0.01,
                    max_value=100.0,
                    step=0.1,
                    key="kfold_gpr_length_scale",
                    help="控制核函数的平滑程度"
                )
                
                # 优化重启次数
                gpr_n_restarts_kfold = st.number_input(
                    "优化重启次数",
                    value=10,
                    min_value=0,
                    step=1,
                    key="kfold_gpr_n_restarts",
                    help="核函数超参数优化的重启次数，越大越可能找到全局最优（注意：次数越多计算时间越长）"
                )
            
            st.subheader("🚀 验证执行")
            
            if st.button("🔄 开始K折交叉验证", type="primary"):
                with st.spinner("正在进行K折交叉验证..."):
                    try:
                        # 构建数据库
                        db = Database(st.session_state.param, selected_snapshots_kfold)
                        
                        # 选择降阶方法
                        if reduction_method_kfold == "POD":
                            reducer = POD()
                        elif reduction_method_kfold == "PODAE":
                            reducer = PODAE()
                        elif reduction_method_kfold == "AE":
                            reducer = AE()
                        
                        # 选择近似方法
                        if approximation_method_kfold == "RBF":
                            approximator = RBF(kernel=rbf_kernel_kfold, epsilon=rbf_epsilon_kfold)
                        elif approximation_method_kfold == "GPR":
                            # 构建GPR核函数
                            from sklearn.gaussian_process.kernels import RBF as RBFGP, Matern, RationalQuadratic, ExpSineSquared, DotProduct, WhiteKernel, ConstantKernel
                            
                            if gpr_kernel_type_kfold == "RBF":
                                kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * RBFGP(
                                    length_scale=gpr_length_scale_kfold, 
                                    length_scale_bounds=(1e-5, 1e5)
                                )
                            elif gpr_kernel_type_kfold == "Matern":
                                kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * Matern(
                                    length_scale=gpr_length_scale_kfold,
                                    length_scale_bounds=(1e-5, 1e5),
                                    nu=matern_nu_kfold
                                )
                            elif gpr_kernel_type_kfold == "RationalQuadratic":
                                kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * RationalQuadratic(
                                    length_scale=gpr_length_scale_kfold,
                                    length_scale_bounds=(1e-5, 1e5)
                                )
                            elif gpr_kernel_type_kfold == "ExpSineSquared":
                                kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * ExpSineSquared(
                                    length_scale=gpr_length_scale_kfold,
                                    length_scale_bounds=(1e-5, 1e5)
                                )
                            elif gpr_kernel_type_kfold == "DotProduct":
                                kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * DotProduct()
                            elif gpr_kernel_type_kfold == "WhiteKernel+RBF":
                                kernel = ConstantKernel(1.0, constant_value_bounds=(1e-5, 1e5)) * RBFGP(
                                    length_scale=gpr_length_scale_kfold,
                                    length_scale_bounds=(1e-5, 1e5)
                                ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-10, 1e1))
                            
                            approximator = GPR(
                                kern=kernel,
                                normalizer=False,
                                optimization_restart=gpr_n_restarts_kfold
                            )
                        elif approximation_method_kfold == "ANN":
                            approximator = ANN()
                        elif approximation_method_kfold == "KNeighborsRegressor":
                            approximator = KNeighborsRegressor()
                        
                        # 构建ROM模型
                        rom = ROM(db, reducer, approximator)
                        rom.fit()
                        
                        # 执行K折交叉验证
                        errors = rom.kfold_cv_error(n_splits=k_value)
                        
                        # 保存结果
                        st.session_state.kfold_results = {
                            'errors': errors,
                            'k_value': k_value,
                            'snapshot_type': selected_snapshot_type_kfold,
                            'reduction_method': reduction_method_kfold,
                            'approximation_method': approximation_method_kfold,
                            'rbf_kernel': rbf_kernel_kfold if approximation_method_kfold == "RBF" else None,
                            'rbf_epsilon': rbf_epsilon_kfold if approximation_method_kfold == "RBF" else None
                        }
                        
                        st.success("✅ K折交叉验证完成!")
                        
                    except Exception as e:
                        st.error(f"❌ K折交叉验证失败: {str(e)}")
        
        # 显示K折验证结果
        if hasattr(st.session_state, 'kfold_results'):
            st.markdown("---")
            st.subheader("📊 K折交叉验证结果")
            
            results = st.session_state.kfold_results
            errors = results['errors']
            
            # 结果概览
            st.info(f"📋 验证配置: {results['snapshot_type']} | {results['reduction_method']} + {results['approximation_method']} | K={results['k_value']}")
            
            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
            with col_metrics1:
                st.metric("平均误差", f"{np.mean(errors):.2e}")
            with col_metrics2:
                st.metric("最大误差", f"{np.max(errors):.2e}")
            with col_metrics3:
                st.metric("最小误差", f"{np.min(errors):.2e}")
            
            # 创建柱状图
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
            plt.rcParams['axes.unicode_minus'] = False    # 用于显示负号
            
            fig, ax = plt.subplots(figsize=(12, 6))
            x_positions = np.arange(len(errors)) + 1
            bars = ax.bar(x_positions, errors, color='skyblue', alpha=0.7)
            ax.axhline(y=np.mean(errors), color='red', linestyle='--', label=f'Mean Error: {np.mean(errors):.2e}')
            
            # 在每个柱子上标注具体的误差值
            for i, error in enumerate(errors):
                ax.text(i+1, error, f'{error:.2e}', ha='center', va='bottom', fontsize=9)
            
            ax.set_xlabel('Fold Number')
            ax.set_ylabel('Error')
            ax.set_title(f'K-fold Cross Validation Errors - {results["approximation_method"]}')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.legend()
            ax.set_xticks(x_positions)
            
            st.pyplot(fig)
            
            # 保存图表到session state
            plot_info = {
                'type': 'kfold_validation',
                'title': f'K-fold Cross Validation - {results["approximation_method"]}',
                'figure': fig,
                'config': results,
                'errors': errors
            }
            if 'generated_plots' not in st.session_state:
                st.session_state.generated_plots = []
            st.session_state.generated_plots.append(plot_info)

# 页面3：联合降阶模型测试
elif page == "🔗 联合降阶模型测试":
    st.title("🔗 联合降阶模型测试")
    
    # 添加页面描述
    st.markdown("""
    <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
        <h4 style='margin: 0 0 10px 0;'>📌 功能说明</h4>
        <p style='margin: 0;'>比较不同降维方法（POD、PODAE、AE）与映射方法（RBF、GPR、ANN等）的组合性能，找出最佳模型配置。</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not EZYRB_AVAILABLE:
        st.error("❌ EZyRB库未安装，无法使用联合降阶模型测试功能")
        st.info("请安装EZyRB库：pip install ezyrb")
        st.stop()
    
    # 检查是否有可用数据
    available_data = {}
    if st.session_state.snapshots_x is not None:
        available_data["X分量"] = st.session_state.snapshots_x
    if st.session_state.snapshots_y is not None:
        available_data["Y分量"] = st.session_state.snapshots_y
    if st.session_state.snapshots_z is not None:
        available_data["Z分量"] = st.session_state.snapshots_z
    if st.session_state.snapshots_stress is not None:
        available_data["应力"] = st.session_state.snapshots_stress
    
    if not available_data or st.session_state.param is None:
        st.warning("⚠️ 没有可用的数据进行联合降阶模型测试，请先在'数据导入与保存'页面加载数据")
        st.stop()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📊 数据选择")
        
        # 选择快照数据
        selected_snapshot_type = st.selectbox(
            "选择快照数据类型",
            list(available_data.keys()),
            key="combined_snapshot_type"
        )
        selected_snapshots = available_data[selected_snapshot_type]
        
        st.info(f"已选择: {selected_snapshot_type} - 形状: {selected_snapshots.shape}")
        st.info(f"参数数据形状: {st.session_state.param.shape}")
        
        st.header("🔧 降维方法选择")
        
        # 可用的降维方法
        reduction_methods = st.multiselect(
            "选择要比较的降维方法",
            ["POD", "PODAE", "AE"],
            default=["POD"],
            help="选择一个或多个降维方法进行比较"
        )
        
        if not reduction_methods:
            st.warning("⚠️ 请至少选择一个降维方法")
    
    with col2:
        st.header("🔗 映射方法选择")
        
        # 可用的映射方法
        mapping_methods = st.multiselect(
            "选择要比较的映射方法",
            ["RBF", "GPR", "KNeighborsRegressor", "RadiusNeighborsRegressor", "ANN"],
            default=["RBF", "GPR"],
            help="选择一个或多个映射方法进行比较"
        )
        
        if not mapping_methods:
            st.warning("⚠️ 请至少选择一个映射方法")
        
        # RBF参数设置
        if "RBF" in mapping_methods:
            st.subheader("RBF参数设置")
            rbf_kernel_combined = st.selectbox(
                "RBF核函数",
                ["multiquadric", "inverse", "gaussian", "linear", "cubic", "quintic", "thin_plate"],
                key="combined_rbf_kernel"
            )
            rbf_epsilon_combined = st.number_input(
                "RBF epsilon", 
                value=0.02, 
                min_value=0.001, 
                max_value=1.0, 
                step=0.001,
                key="combined_rbf_epsilon"
            )
        
        # GPR参数设置
        if "GPR" in mapping_methods:
            st.subheader("GPR参数设置")
            gpr_kernel_type_combined = st.selectbox(
                "GPR核函数类型",
                ["Matern", "RBF", "RationalQuadratic"],
                key="combined_gpr_kernel_type",
                help="选择高斯过程的核函数类型"
            )
            
            if gpr_kernel_type_combined == "Matern":
                matern_nu_combined = st.selectbox(
                    "Matern nu参数",
                    [0.5, 1.5, 2.5],
                    index=1,
                    key="combined_matern_nu"
                )
            
            gpr_n_restarts_combined = st.number_input(
                "优化重启次数",
                value=10,
                min_value=0,
                step=1,
                key="combined_gpr_n_restarts",
                help="核函数超参数优化的重启次数，越大越可能找到全局最优（注意：次数越多计算时间越长）"
            )
        
        st.header("⚙️ 测试设置")
        
        # K折设置
        max_k_combined = len(st.session_state.param)
        k_value_combined = st.slider(
            "K折交叉验证折数",
            min_value=2,
            max_value=max_k_combined,
            value=min(7, max_k_combined),
            help=f"设置K折交叉验证的折数（最大可设为参数点数：{max_k_combined}）"
        )
    
    # 执行测试按钮
    if reduction_methods and mapping_methods:
        if st.button("🚀 开始联合模型测试", type="primary"):
            with st.spinner("正在进行联合降阶模型测试..."):
                try:
                    # 准备数据
                    param_data = st.session_state.param
                    snapshot_data = selected_snapshots
                    
                    # 创建数据库
                    db = Database(param_data, snapshot_data)
                    
                    # 存储性能数据
                    performance_data = {
                        'errors': {},
                        'fit_times': {},
                        'prediction_times': {},
                        'memory_usage': {}
                    }
                    
                    # 进度条
                    total_combinations = len(reduction_methods) * len(mapping_methods)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    current_progress = 0
                    
                    # 测试每个组合
                    for red_method in reduction_methods:
                        performance_data['errors'][red_method] = {}
                        performance_data['fit_times'][red_method] = {}
                        performance_data['prediction_times'][red_method] = {}
                        
                        for map_method in mapping_methods:
                            status_text.text(f"测试组合: {red_method} + {map_method}")
                            
                            try:
                                # 创建降维器
                                if red_method == "POD":
                                    reducer = POD()
                                elif red_method == "PODAE":
                                    reducer = PODAE()
                                elif red_method == "AE":
                                    reducer = AE()
                                
                                # 创建映射器
                                if map_method == "RBF":
                                    approximator = RBF(kernel=rbf_kernel_combined, epsilon=rbf_epsilon_combined)
                                elif map_method == "GPR":
                                    from sklearn.gaussian_process.kernels import RBF as RBFGP, Matern, RationalQuadratic, ConstantKernel
                                    
                                    if gpr_kernel_type_combined == "RBF":
                                        kernel = ConstantKernel(1.0) * RBFGP(length_scale=1.0)
                                    elif gpr_kernel_type_combined == "Matern":
                                        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=matern_nu_combined)
                                    elif gpr_kernel_type_combined == "RationalQuadratic":
                                        kernel = ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0)
                                    
                                    approximator = GPR(kern=kernel, normalizer=False, optimization_restart=gpr_n_restarts_combined)
                                elif map_method == "KNeighborsRegressor":
                                    approximator = KNeighborsRegressor(n_neighbors=5, weights='distance')
                                elif map_method == "RadiusNeighborsRegressor":
                                    approximator = RadiusNeighborsRegressor(radius=1.0, weights='distance')
                                elif map_method == "ANN":
                                    approximator = ANN([6, 12, 24], function=nn.ReLU(), stop_training=[1000, 1e-8])
                                
                                # 创建ROM模型
                                rom = ROM(db, reducer, approximator)
                                
                                # 测量训练时间
                                import time
                                fit_start = time.time()
                                rom.fit()
                                fit_time = time.time() - fit_start
                                
                                # K折交叉验证
                                errors = rom.kfold_cv_error(n_splits=k_value_combined)
                                avg_error = np.mean(errors)
                                
                                # 存储结果
                                performance_data['errors'][red_method][map_method] = avg_error
                                performance_data['fit_times'][red_method][map_method] = fit_time
                                
                            except Exception as e:
                                st.warning(f"⚠️ {red_method} + {map_method} 测试失败: {str(e)}")
                                performance_data['errors'][red_method][map_method] = np.nan
                                performance_data['fit_times'][red_method][map_method] = np.nan
                            
                            current_progress += 1
                            progress_bar.progress(current_progress / total_combinations)
                    
                    status_text.text("✅ 测试完成!")
                    
                    # 保存结果到session state
                    st.session_state.combined_test_results = {
                        'performance_data': performance_data,
                        'reduction_methods': reduction_methods,
                        'mapping_methods': mapping_methods,
                        'snapshot_type': selected_snapshot_type,
                        'k_value': k_value_combined
                    }
                    
                    st.success("✅ 联合降阶模型测试完成!")
                    
                except Exception as e:
                    st.error(f"❌ 测试失败: {str(e)}")
    
    # 显示结果
    if hasattr(st.session_state, 'combined_test_results'):
        st.markdown("---")
        st.header("📊 测试结果")
        
        results = st.session_state.combined_test_results
        performance_data = results['performance_data']
        
        # 创建结果表格
        st.subheader("📋 K折交叉验证误差")
        
        # 准备数据框
        error_data = []
        for red_method in results['reduction_methods']:
            row_data = {'降维方法': red_method}
            for map_method in results['mapping_methods']:
                if map_method in performance_data['errors'].get(red_method, {}):
                    error_value = performance_data['errors'][red_method][map_method]
                    row_data[map_method] = f"{error_value:.4e}" if not np.isnan(error_value) else "N/A"
                else:
                    row_data[map_method] = "N/A"
            error_data.append(row_data)
        
        error_df = pd.DataFrame(error_data)
        st.dataframe(error_df, use_container_width=True)
        
        # 创建可视化图表
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # 误差热力图
            st.subheader("📊 误差热力图")
            
            # 准备热力图数据
            heatmap_data = []
            for red_method in results['reduction_methods']:
                row = []
                for map_method in results['mapping_methods']:
                    if map_method in performance_data['errors'].get(red_method, {}):
                        value = performance_data['errors'][red_method][map_method]
                        row.append(value if not np.isnan(value) else 0)
                    else:
                        row.append(0)
                heatmap_data.append(row)
            
            if heatmap_data:
                # 设置中文字体
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
                plt.rcParams['axes.unicode_minus'] = False    # 用于显示负号
                
                fig_heatmap, ax_heatmap = plt.subplots(figsize=(10, 6))
                im = ax_heatmap.imshow(heatmap_data, cmap='RdYlGn_r', aspect='auto')
                
                # 设置标签
                ax_heatmap.set_xticks(np.arange(len(results['mapping_methods'])))
                ax_heatmap.set_yticks(np.arange(len(results['reduction_methods'])))
                ax_heatmap.set_xticklabels(results['mapping_methods'])
                ax_heatmap.set_yticklabels(results['reduction_methods'])
                
                # 旋转x轴标签
                plt.setp(ax_heatmap.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                
                # 添加颜色条
                cbar = plt.colorbar(im, ax=ax_heatmap)
                cbar.set_label('K-fold CV Error')
                
                # 添加数值标注
                for i in range(len(results['reduction_methods'])):
                    for j in range(len(results['mapping_methods'])):
                        text = ax_heatmap.text(j, i, f'{heatmap_data[i][j]:.2e}',
                                             ha="center", va="center", color="black", fontsize=9)
                
                ax_heatmap.set_title(f'联合模型误差热力图 (K={k_value_combined})')
                plt.tight_layout()
                st.pyplot(fig_heatmap)
        
        with col_chart2:
            # 训练时间对比
            st.subheader("⏱️ 训练时间对比")
            
            # 准备柱状图数据
            labels = []
            times = []
            colors = []
            color_map = plt.cm.Set3(np.linspace(0, 1, len(results['reduction_methods'])))
            
            for i, red_method in enumerate(results['reduction_methods']):
                for map_method in results['mapping_methods']:
                    if map_method in performance_data['fit_times'].get(red_method, {}):
                        time_value = performance_data['fit_times'][red_method][map_method]
                        if not np.isnan(time_value):
                            labels.append(f"{red_method}\n{map_method}")
                            times.append(time_value)
                            colors.append(color_map[i])
            
            if times:
                # 设置中文字体
                plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文
                plt.rcParams['axes.unicode_minus'] = False    # 用于显示负号
                
                fig_time, ax_time = plt.subplots(figsize=(10, 6))
                bars = ax_time.bar(range(len(labels)), times, color=colors, alpha=0.7)
                ax_time.set_xlabel('模型组合')
                ax_time.set_ylabel('训练时间 (秒)')
                ax_time.set_title('不同模型组合的训练时间')
                ax_time.set_xticks(range(len(labels)))
                ax_time.set_xticklabels(labels, rotation=45, ha='right')
                
                # 添加数值标注
                for bar, time in zip(bars, times):
                    height = bar.get_height()
                    ax_time.text(bar.get_x() + bar.get_width()/2., height,
                               f'{time:.2f}', ha='center', va='bottom')
                
                plt.tight_layout()
                st.pyplot(fig_time)
        
        # 最佳组合推荐
        st.subheader("🏆 最佳组合推荐")
        
        # 找出最小误差的组合
        min_error = float('inf')
        best_combination = None
        
        for red_method in results['reduction_methods']:
            for map_method in results['mapping_methods']:
                if map_method in performance_data['errors'].get(red_method, {}):
                    error = performance_data['errors'][red_method][map_method]
                    if not np.isnan(error) and error < min_error:
                        min_error = error
                        best_combination = (red_method, map_method)
        
        if best_combination:
            col_best1, col_best2, col_best3 = st.columns(3)
            with col_best1:
                st.metric("最佳降维方法", best_combination[0])
            with col_best2:
                st.metric("最佳映射方法", best_combination[1])
            with col_best3:
                st.metric("最小误差", f"{min_error:.4e}")
        
        # 保存图表
        if 'generated_plots' not in st.session_state:
            st.session_state.generated_plots = []
        
        plot_info = {
            'type': 'combined_model_test',
            'title': f'Combined Model Test - {len(results["reduction_methods"])}×{len(results["mapping_methods"])} combinations',
            'figures': [fig_heatmap, fig_time] if 'fig_heatmap' in locals() and 'fig_time' in locals() else [],
            'config': results,
            'best_combination': best_combination if 'best_combination' in locals() else None
        }
        st.session_state.generated_plots.append(plot_info)

# 页面4：三维可视化
elif page == "🎨 三维可视化":
    st.title("🎨 三维可视化")
    
    # 添加页面描述
    st.markdown("""
    <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
        <h4 style='margin: 0 0 10px 0;'>📌 功能说明</h4>
        <p style='margin: 0;'>提供原始网格、形变对比和预测误差的三维可视化功能。支持交互式窗口和静态图像两种模式。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 检测云环境并显示警告
    if is_cloud_environment():
        st.warning("⚠️ 检测到云环境：交互式3D窗口可能不可用，系统将自动切换到静态图像模式")
        st.info("💡 如需完整的交互式体验，请在本地环境中运行此应用")
    
    # 检查是否有网格数据
    if st.session_state.mesh_data is None:
        st.warning("⚠️ 没有可用的网格数据，请先在'数据导入与保存'页面加载VTU文件")
        st.info("💡 三维可视化需要VTU文件中的网格数据")
        st.stop()
    
    # 显示网格信息
    if st.session_state.mesh_info:
        st.info(st.session_state.mesh_info)
    
    # 创建三个选项卡
    tab1, tab2, tab3 = st.tabs(["🔷 原始图", "🔄 形变对比图", "📊 预测误差图"])
    
    with tab1:
        st.header("🔷 原始网格可视化")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("⚙️ 可视化设置")
            
            # 透明度设置
            opacity_original = st.slider(
                "透明度",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.1,
                key="opacity_original"
            )
            
            # 颜色映射选择
            cmap_original = st.selectbox(
                "颜色映射",
                ["viridis", "plasma", "inferno", "magma", "cividis", "rainbow", "jet", "coolwarm"],
                key="cmap_original"
            )
            
            # 显示边缘
            show_edges_original = st.checkbox("显示边缘", value=True, key="edges_original")
            
            # 视角选择
            view_option = st.selectbox(
                "视角",
                ["等轴测视图", "XY平面", "XZ平面", "YZ平面"],
                key="view_original"
            )
            
            # 选择要显示的数据
            available_arrays = list(st.session_state.mesh_data.array_names)
            if available_arrays:
                selected_array = st.selectbox(
                    "选择要显示的数据",
                    available_arrays,
                    key="array_original"
                )
            else:
                selected_array = None
                st.warning("⚠️ 网格中没有可用的数据数组")
        
        with col1:
            # 可视化模式选择
            viz_mode = st.radio(
                "可视化模式",
                ["交互式窗口", "静态图像"],
                key="viz_mode_original",
                help="交互式窗口：可旋转缩放，但会打开新窗口；静态图像：嵌入页面，但不可交互"
            )
            
            if st.button("🎨 生成原始图", type="primary", key="btn_original"):
                with st.spinner("正在生成三维可视化..."):
                    try:
                        # 复制网格数据
                        mesh = st.session_state.mesh_data.copy()
                        
                        # 准备可视化参数
                        viz_kwargs = {
                            'scalars': selected_array if selected_array and selected_array in mesh.array_names else None,
                            'cmap': cmap_original,
                            'opacity': opacity_original,
                            'show_edges': show_edges_original,
                            'title': "原始网格可视化"
                        }
                        
                        if viz_mode == "交互式窗口" and not is_cloud_environment():
                            # 本地环境尝试交互式窗口
                            def create_interactive_plot():
                                import pyvista as pv
                                pv.set_plot_theme("document")
                                
                                plotter = pv.Plotter(window_size=[800, 600])
                                
                                # 添加网格
                                if viz_kwargs['scalars']:
                                    mesh.set_active_scalars(viz_kwargs['scalars'])
                                    plotter.add_mesh(
                                        mesh,
                                        scalars=viz_kwargs['scalars'],
                                        opacity=viz_kwargs['opacity'],
                                        cmap=viz_kwargs['cmap'],
                                        show_edges=viz_kwargs['show_edges'],
                                        edge_color='black',
                                        show_scalar_bar=True
                                    )
                                else:
                                    plotter.add_mesh(
                                        mesh,
                                        color='lightgray',
                                        opacity=viz_kwargs['opacity'],
                                        show_edges=viz_kwargs['show_edges'],
                                        edge_color='black'
                                    )
                                
                                # 设置视角
                                if view_option == "等轴测视图":
                                    plotter.view_isometric()
                                elif view_option == "XY平面":
                                    plotter.view_xy()
                                elif view_option == "XZ平面":
                                    plotter.view_xz()
                                elif view_option == "YZ平面":
                                    plotter.view_yz()
                                
                                plotter.add_axes()
                                
                                # 显示交互式窗口
                                st.info("🖱️ 交互式窗口已打开，您可以：\n• 左键拖动旋转\n• 右键拖动平移\n• 滚轮缩放\n• 关闭窗口后继续")
                                plotter.show()
                                return True
                            
                            def create_fallback_plot():
                                # 设置标量数据
                                if viz_kwargs['scalars']:
                                    scalars_data = mesh.get_array(viz_kwargs['scalars'])
                                else:
                                    scalars_data = None
                                
                                return create_cloud_friendly_plot(
                                    mesh, 
                                    scalars=scalars_data,
                                    cmap=viz_kwargs['cmap'],
                                    opacity=viz_kwargs['opacity'],
                                    show_edges=viz_kwargs['show_edges'],
                                    title=viz_kwargs['title']
                                )
                            
                            # 使用安全的交互式窗口函数
                            result, error_msg = create_safe_interactive_window(create_interactive_plot, create_fallback_plot)
                            
                            if error_msg:
                                st.warning(f"⚠️ {error_msg}")
                                if result:
                                    # 显示备选方案的结果
                                    image, method = result
                                    if isinstance(image, bytes):
                                        st.image(image, caption=f"{viz_kwargs['title']} ({method})", use_column_width=True)
                                    else:
                                        st.image(image, caption=f"{viz_kwargs['title']} ({method})", use_column_width=True)
                                    st.success(f"✅ 使用 {method} 成功生成可视化图像")
                                else:
                                    viz_mode = "静态图像"  # 强制切换到静态模式
                        
                        if viz_mode == "静态图像" or is_cloud_environment():
                            # 使用云环境友好的可视化函数
                            try:
                                # 设置标量数据
                                if viz_kwargs['scalars']:
                                    scalars_data = mesh.get_array(viz_kwargs['scalars'])
                                else:
                                    scalars_data = None
                                
                                image, method = create_cloud_friendly_plot(
                                    mesh, 
                                    scalars=scalars_data,
                                    cmap=viz_kwargs['cmap'],
                                    opacity=viz_kwargs['opacity'],
                                    show_edges=viz_kwargs['show_edges'],
                                    title=viz_kwargs['title']
                                )
                                
                                # 显示图像
                                if isinstance(image, bytes):
                                    st.image(image, caption=f"{viz_kwargs['title']} ({method})", use_column_width=True)
                                else:
                                    st.image(image, caption=f"{viz_kwargs['title']} ({method})", use_column_width=True)
                                
                                st.success(f"✅ 使用 {method} 成功生成可视化图像")
                                
                            except Exception as fallback_error:
                                st.error(f"❌ 所有可视化方法都失败了: {str(fallback_error)}")
                                st.info("💡 建议：尝试在本地环境运行以获得完整的3D可视化功能")
                        
                        # 显示网格统计信息
                        if selected_array:
                            scalar_data = mesh.get_array(selected_array)
                            st.info(f"""
                            📊 数据统计 ({selected_array}):
                            • 最大值: {scalar_data.max():.6f}
                            • 最小值: {scalar_data.min():.6f}
                            • 平均值: {scalar_data.mean():.6f}
                            • 标准差: {scalar_data.std():.6f}
                            """)
                        
                    except Exception as e:
                        st.error(f"❌ 可视化失败: {str(e)}")
                        st.info("💡 提示：请确保已安装PyVista: pip install pyvista")
    
    with tab2:
        st.header("🔄 形变对比图")
        
        # 检查是否有位移数据
        has_displacement = (st.session_state.snapshots_x is not None and 
                          st.session_state.snapshots_y is not None and 
                          st.session_state.snapshots_z is not None)
        
        if not has_displacement:
            st.warning("⚠️ 没有位移数据，无法显示形变对比")
            st.info("💡 请确保已加载X、Y、Z三个方向的位移数据")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.subheader("⚙️ 形变设置")
                
                # 选择时间步
                max_timesteps = len(st.session_state.snapshots_x)
                timestep = st.slider(
                    "选择时间步",
                    min_value=0,
                    max_value=max_timesteps-1,
                    value=0,
                    key="timestep_deform"
                )
                
                # 形变放大系数
                deform_factor = st.number_input(
                    "形变放大系数",
                    min_value=0.1,
                    max_value=100.0,
                    value=10.0,
                    step=0.1,
                    key="deform_factor"
                )
                
                # 透明度设置
                opacity_deform = st.slider(
                    "形变网格透明度",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    key="opacity_deform"
                )
                
                # 颜色映射
                cmap_deform = st.selectbox(
                    "颜色映射",
                    ["plasma", "viridis", "inferno", "magma", "rainbow", "jet", "coolwarm"],
                    key="cmap_deform"
                )
                
                # 显示原始网格
                show_original = st.checkbox("显示原始网格", value=True, key="show_original_deform")
                
                # 视角选择
                view_option_deform = st.selectbox(
                    "视角",
                    ["等轴测视图", "XY平面", "XZ平面", "YZ平面"],
                    key="view_deform"
                )
            
            with col1:
                # 可视化模式选择
                viz_mode_deform = st.radio(
                    "可视化模式",
                    ["交互式窗口", "静态图像"],
                    key="viz_mode_deform",
                    help="交互式窗口：可旋转缩放，但会打开新窗口；静态图像：嵌入页面，但不可交互"
                )
                
                if st.button("🎨 生成形变对比图", type="primary", key="btn_deform"):
                    with st.spinner("正在生成形变对比图..."):
                        try:
                            import pyvista as pv
                            pv.set_plot_theme("document")
                            
                            # 复制网格
                            mesh = st.session_state.mesh_data.copy()
                            
                            # 获取位移数据
                            u = st.session_state.snapshots_x[timestep]
                            v = st.session_state.snapshots_y[timestep]
                            w = st.session_state.snapshots_z[timestep]
                            
                            # 创建位移向量
                            displacement = np.column_stack((u, v, w))
                            mesh["displacement"] = displacement
                            
                            # 计算位移大小
                            displacement_magnitude = np.sqrt(u**2 + v**2 + w**2)
                            mesh["displacement_magnitude"] = displacement_magnitude
                            
                            # 创建变形网格
                            warped = mesh.copy()
                            warped = warped.warp_by_vector("displacement", factor=deform_factor)
                            
                            if viz_mode_deform == "交互式窗口":
                                # 定义交互式绘图函数
                                def create_deform_interactive():
                                    # 尝试创建交互式绘图器
                                    plotter = pv.Plotter(window_size=[800, 600])
                                    
                                    # 显示原始网格
                                    if show_original:
                                        plotter.add_mesh(
                                            mesh,
                                            color="gray",
                                            opacity=0.3,
                                            show_edges=True,
                                            edge_color='black',
                                            label="Original"
                                        )
                                    
                                    # 显示变形网格
                                    plotter.add_mesh(
                                        warped,
                                        scalars="displacement_magnitude",
                                        opacity=opacity_deform,
                                        cmap=cmap_deform,
                                        show_edges=True,
                                        edge_color='black',
                                        label=f"Deformed (×{deform_factor})",
                                        show_scalar_bar=True,
                                        scalar_bar_args={"title": "Displacement"}
                                    )
                                    
                                    # 设置视角和其他元素
                                    plotter.add_legend()
                                    
                                    # 设置视角
                                    if view_option_deform == "等轴测视图":
                                        plotter.view_isometric()
                                    elif view_option_deform == "XY平面":
                                        plotter.view_xy()
                                    elif view_option_deform == "XZ平面":
                                        plotter.view_xz()
                                    elif view_option_deform == "YZ平面":
                                        plotter.view_yz()
                                    
                                    plotter.add_axes()
                                    
                                    # 显示交互式窗口
                                    st.info("🖱️ 交互式窗口已打开，您可以：\n• 左键拖动旋转\n• 右键拖动平移\n• 滚轮缩放\n• 关闭窗口后继续")
                                    plotter.show()
                                    return True
                                
                                def create_deform_fallback():
                                    # 创建离屏绘图器用于静态图像
                                    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
                                    
                                    # 显示原始网格
                                    if show_original:
                                        plotter.add_mesh(
                                            mesh,
                                            color="gray",
                                            opacity=0.3,
                                            show_edges=True,
                                            edge_color='black',
                                            label="Original"
                                        )
                                    
                                    # 显示变形网格
                                    plotter.add_mesh(
                                        warped,
                                        scalars="displacement_magnitude",
                                        opacity=opacity_deform,
                                        cmap=cmap_deform,
                                        show_edges=True,
                                        edge_color='black',
                                        label=f"Deformed (×{deform_factor})",
                                        show_scalar_bar=True,
                                        scalar_bar_args={"title": "Displacement"}
                                    )
                                    
                                    # 设置视角和其他元素
                                    plotter.add_legend()
                                    
                                    # 设置视角
                                    if view_option_deform == "等轴测视图":
                                        plotter.view_isometric()
                                    elif view_option_deform == "XY平面":
                                        plotter.view_xy()
                                    elif view_option_deform == "XZ平面":
                                        plotter.view_xz()
                                    elif view_option_deform == "YZ平面":
                                        plotter.view_yz()
                                    
                                    plotter.add_axes()
                                    
                                    # 截图并显示
                                    plotter.show(auto_close=False)
                                    image = plotter.screenshot()
                                    plotter.close()
                                    return image
                                
                                # 使用安全的交互式窗口函数
                                result, error_msg = create_safe_interactive_window(create_deform_interactive, create_deform_fallback)
                                
                                if error_msg and result:
                                    st.warning(f"⚠️ {error_msg}")
                                    # 显示备选方案的结果
                                    st.image(result, caption=f"形变对比图 (静态模式, 放大系数: {deform_factor})", use_column_width=True)
                                elif error_msg:
                                    st.error(f"❌ 交互式和备选方案都失败了: {error_msg}")
                                    viz_mode_deform = "静态图像"  # 强制切换到静态模式
                                
                            if viz_mode_deform == "静态图像" or is_cloud_environment():
                                # 使用云环境友好的可视化函数
                                try:
                                    # 为形变网格创建一个组合可视化
                                    image, method = create_cloud_friendly_plot(
                                        warped,
                                        scalars=displacement_magnitude,
                                        cmap=cmap_deform,
                                        opacity=opacity_deform,
                                        show_edges=True,
                                        title=f"形变对比图 (×{deform_factor})"
                                    )
                                    
                                    # 显示图像
                                    if isinstance(image, bytes):
                                        st.image(image, caption=f"形变对比图 ({method}, 放大系数: {deform_factor})", use_column_width=True)
                                    else:
                                        st.image(image, caption=f"形变对比图 ({method}, 放大系数: {deform_factor})", use_column_width=True)
                                    
                                    st.success(f"✅ 使用 {method} 成功生成形变对比图")
                                    
                                except Exception as fallback_error:
                                    st.error(f"❌ 所有形变可视化方法都失败了: {str(fallback_error)}")
                                    st.info("💡 建议：尝试在本地环境运行以获得完整的3D可视化功能")
                            
                            # 显示统计信息
                            st.info(f"""
                            📊 位移统计 (时间步 {timestep + 1}/{max_timesteps}):
                            • 最大位移: {displacement_magnitude.max():.6f}
                            • 最小位移: {displacement_magnitude.min():.6f}
                            • 平均位移: {displacement_magnitude.mean():.6f}
                            • 标准差: {displacement_magnitude.std():.6f}
                            """)
                            
                            # 参数信息
                            if st.session_state.param is not None and timestep < len(st.session_state.param):
                                st.info(f"📌 参数值: {st.session_state.param[timestep].flatten()[0]:.2f}")
                            
                        except Exception as e:
                            st.error(f"❌ 形变可视化失败: {str(e)}")
                            st.info("💡 提示：请确保已加载X、Y、Z三个方向的位移数据")
    
    with tab3:
        st.header("📊 预测误差图")
        
        # 检查是否有预测结果
        if not hasattr(st.session_state, 'prediction_results'):
            st.warning("⚠️ 没有预测结果，请先在'预测测试'页面进行预测")
        else:
            col1, col2 = st.columns([2, 1])
            
            with col2:
                st.subheader("⚙️ 误差显示设置")
                
                # 选择验证点
                results = st.session_state.prediction_results
                validation_points = [f"验证点 {r['validation_idx']+1}" for r in results]
                selected_val_point = st.selectbox(
                    "选择验证点",
                    validation_points,
                    key="val_point_error"
                )
                val_idx = validation_points.index(selected_val_point)
                
                # 误差阈值设置
                error_threshold_method = st.radio(
                    "误差阈值方法",
                    ["标准差", "百分位数", "自定义"],
                    key="error_threshold_method"
                )
                
                if error_threshold_method == "标准差":
                    std_multiplier = st.slider(
                        "标准差倍数",
                        min_value=0.5,
                        max_value=3.0,
                        value=1.0,
                        step=0.1,
                        key="std_multiplier"
                    )
                elif error_threshold_method == "百分位数":
                    percentile = st.slider(
                        "百分位数",
                        min_value=50,
                        max_value=99,
                        value=90,
                        key="percentile"
                    )
                else:
                    custom_threshold = st.number_input(
                        "自定义阈值",
                        min_value=0.0,
                        value=0.1,
                        step=0.01,
                        key="custom_threshold"
                    )
                
                # 颜色设置
                high_error_color = st.color_picker("高误差颜色", "#FF0000", key="high_error_color")
                low_error_color = st.color_picker("低误差颜色", "#0000FF", key="low_error_color")
                
                # 透明度设置
                high_error_opacity = st.slider(
                    "高误差区域透明度",
                    min_value=0.0,
                    max_value=1.0,
                    value=1.0,
                    step=0.1,
                    key="high_error_opacity"
                )
                
                low_error_opacity = st.slider(
                    "低误差区域透明度",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.3,
                    step=0.1,
                    key="low_error_opacity"
                )
                
                # 显示边缘
                show_edges_error = st.checkbox("显示边缘", value=True, key="edges_error")
            
            with col1:
                # 可视化模式选择
                viz_mode_error = st.radio(
                    "可视化模式",
                    ["交互式窗口", "静态图像"],
                    key="viz_mode_error",
                    help="交互式窗口：可旋转缩放，但会打开新窗口；静态图像：嵌入页面，但不可交互"
                )
                
                if st.button("🎨 生成误差图", type="primary", key="btn_error"):
                    with st.spinner("正在生成预测误差图..."):
                        try:
                            import pyvista as pv
                            pv.set_plot_theme("document")
                            
                            # 获取选中的结果
                            result = results[val_idx]
                            
                            # 计算误差
                            true_snapshot = result['validation_snapshot']
                            predicted_snapshot = result['predicted_snapshot']
                            
                            # 计算相对误差（参考Visualization.py的方法）
                            mean_true = np.mean(np.abs(true_snapshot))
                            if mean_true > 0:
                                error = np.abs(predicted_snapshot - true_snapshot) / mean_true
                            else:
                                error = np.abs(predicted_snapshot - true_snapshot)
                            
                            # 确定阈值
                            if error_threshold_method == "标准差":
                                threshold = np.mean(error) + std_multiplier * np.std(error)
                            elif error_threshold_method == "百分位数":
                                threshold = np.percentile(error, percentile)
                            else:
                                threshold = custom_threshold
                            
                            # 复制网格
                            mesh = st.session_state.mesh_data.copy()
                            
                            # 添加误差数据
                            mesh["error"] = error
                            
                            # 创建颜色数组
                            above_threshold = error > threshold
                            
                            # 转换颜色为RGB
                            high_color_rgb = [int(high_error_color[i:i+2], 16)/255 for i in (1, 3, 5)]
                            low_color_rgb = [int(low_error_color[i:i+2], 16)/255 for i in (1, 3, 5)]
                            
                            # 创建RGBA颜色数组
                            colors = np.zeros((mesh.n_points, 4))
                            colors[above_threshold] = high_color_rgb + [high_error_opacity]
                            colors[~above_threshold] = low_color_rgb + [low_error_opacity]
                            
                            if viz_mode_error == "交互式窗口":
                                # 定义交互式误差图函数
                                def create_error_interactive():
                                    # 尝试创建交互式绘图器
                                    plotter = pv.Plotter(window_size=[800, 600])
                                    
                                    # 添加网格 - 使用两个不同的网格来显示不同颜色
                                    # 首先添加低误差点
                                    if np.any(~above_threshold):
                                        mesh_low = mesh.extract_points(~above_threshold)
                                        plotter.add_mesh(
                                            mesh_low,
                                            color=low_color_rgb,
                                            opacity=low_error_opacity,
                                            show_edges=show_edges_error,
                                            edge_color='black',
                                            label=f"Error < {threshold:.4f}"
                                        )
                                    
                                    # 然后添加高误差点
                                    if np.any(above_threshold):
                                        mesh_high = mesh.extract_points(above_threshold)
                                        plotter.add_mesh(
                                            mesh_high,
                                            color=high_color_rgb,
                                            opacity=high_error_opacity,
                                            show_edges=show_edges_error,
                                            edge_color='black',
                                            label=f"Error > {threshold:.4f}"
                                        )
                                    
                                    # 添加标题和其他元素（使用英文避免中文显示问题）
                                    plotter.add_text(
                                        f"Error Distribution - Point {result['validation_idx']+1}",
                                        position='upper_edge',
                                        font_size=12,
                                        color='black'
                                    )
                                    
                                    plotter.add_legend()
                                    plotter.view_isometric()
                                    plotter.add_axes()
                                    
                                    # 显示交互式窗口
                                    st.info("🖱️ 交互式窗口已打开，您可以：\n• 左键拖动旋转\n• 右键拖动平移\n• 滚轮缩放\n• 关闭窗口后继续")
                                    plotter.show()
                                    return True
                                
                                def create_error_fallback():
                                    # 创建离屏绘图器用于静态图像
                                    plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
                                    
                                    # 添加网格 - 使用两个不同的网格来显示不同颜色
                                    # 首先添加低误差点
                                    if np.any(~above_threshold):
                                        mesh_low = mesh.extract_points(~above_threshold)
                                        plotter.add_mesh(
                                            mesh_low,
                                            color=low_color_rgb,
                                            opacity=low_error_opacity,
                                            show_edges=show_edges_error,
                                            edge_color='black',
                                            label=f"Error < {threshold:.4f}"
                                        )
                                    
                                    # 然后添加高误差点
                                    if np.any(above_threshold):
                                        mesh_high = mesh.extract_points(above_threshold)
                                        plotter.add_mesh(
                                            mesh_high,
                                            color=high_color_rgb,
                                            opacity=high_error_opacity,
                                            show_edges=show_edges_error,
                                            edge_color='black',
                                            label=f"Error > {threshold:.4f}"
                                        )
                                    
                                    # 添加标题和其他元素（使用英文避免中文显示问题）
                                    plotter.add_text(
                                        f"Error Distribution - Point {result['validation_idx']+1}",
                                        position='upper_edge',
                                        font_size=12,
                                        color='black'
                                    )
                                    
                                    plotter.add_legend()
                                    plotter.view_isometric()
                                    plotter.add_axes()
                                    
                                    # 截图并显示
                                    plotter.show(auto_close=False)
                                    image = plotter.screenshot()
                                    plotter.close()
                                    return image
                                
                                # 使用安全的交互式窗口函数
                                result_img, error_msg = create_safe_interactive_window(create_error_interactive, create_error_fallback)
                                
                                if error_msg and result_img:
                                    st.warning(f"⚠️ {error_msg}")
                                    # 显示备选方案的结果
                                    st.image(result_img, caption="预测误差分布图 (静态模式)", use_column_width=True)
                                elif error_msg:
                                    st.error(f"❌ 交互式和备选方案都失败了: {error_msg}")
                                    viz_mode_error = "静态图像"  # 强制切换到静态模式
                                
                            if viz_mode_error == "静态图像" or is_cloud_environment():
                                # 使用云环境友好的可视化函数
                                try:
                                    # 使用误差数据作为标量进行可视化
                                    image, method = create_cloud_friendly_plot(
                                        mesh,
                                        scalars=error,
                                        cmap='RdBu_r',  # 红蓝色图，红色表示高误差
                                        opacity=0.8,
                                        show_edges=show_edges_error,
                                        title=f"预测误差分布 - 验证点 {result['validation_idx']+1}"
                                    )
                                    
                                    # 显示图像
                                    if isinstance(image, bytes):
                                        st.image(image, caption=f"预测误差分布图 ({method})", use_column_width=True)
                                    else:
                                        st.image(image, caption=f"预测误差分布图 ({method})", use_column_width=True)
                                    
                                    st.success(f"✅ 使用 {method} 成功生成误差分布图")
                                    
                                except Exception as fallback_error:
                                    st.error(f"❌ 所有误差可视化方法都失败了: {str(fallback_error)}")
                                    st.info("💡 建议：尝试在本地环境运行以获得完整的3D可视化功能")
                            
                            # 显示误差统计
                            st.info(f"""
                            📊 误差统计:
                            • 最大相对误差: {error.max():.4f}
                            • 最小相对误差: {error.min():.4f}
                            • 平均相对误差: {error.mean():.4f}
                            • 误差标准差: {error.std():.4f}
                            • 设定阈值: {threshold:.4f}
                            • 超过阈值的点数: {np.sum(above_threshold)} / {len(error)} ({np.sum(above_threshold)/len(error)*100:.1f}%)
                            """)
                            
                            # 参数信息
                            if st.session_state.param is not None:
                                st.info(f"📌 验证参数值: {result['validation_param'].flatten()[0]:.2f}")
                            
                        except Exception as e:
                            st.error(f"❌ 误差可视化失败: {str(e)}")
                            st.info("💡 提示：请确保已进行预测测试并有可用的结果")

# 页面5：图表输出
elif page == "📈 图表输出":
    st.title("📈 图表输出")
    
    # 添加页面描述
    st.markdown("""
    <div style='background-color: #e8f4f8; padding: 20px; border-radius: 10px; margin-bottom: 30px;'>
        <h4 style='margin: 0 0 10px 0;'>📌 功能说明</h4>
        <p style='margin: 0;'>查看和管理所有生成的分析图表，支持批量导出和清除功能。</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'generated_plots' not in st.session_state or not st.session_state.generated_plots:
        st.info("ℹ️ 还没有生成任何图表，请先在'预测测试'页面进行测试")
    else:
        st.subheader("📊 已生成的图表")
        
        # 显示图表列表
        for i, plot_info in enumerate(st.session_state.generated_plots):
            with st.expander(f"图表 {i+1}: {plot_info['title']}"):
                st.write(f"**类型**: {plot_info['type']}")
                
                # 显示配置信息
                config = plot_info['config']
                if isinstance(config, dict):
                    config_str = f"{config.get('snapshot_type', 'N/A')} | {config.get('reduction_method', 'N/A')} + {config.get('approximation_method', 'N/A')}"
                    if 'validation_mode' in config:
                        config_str += f" | {config['validation_mode']}"
                    st.write(f"**配置**: {config_str}")
                else:
                    st.write(f"**配置**: {config}")
                
                if plot_info['type'] == 'parameter_prediction_single':
                    st.write("**包含图表**: 平均值对比图、最大误差点对比图、点对点对比图")
                    st.write(f"**验证点**: {plot_info['validation_idx'] + 1}")
                    for j, fig in enumerate(plot_info['figures']):
                        st.pyplot(fig)
                
                elif plot_info['type'] == 'parameter_prediction_multi':
                    st.write("**包含图表**: 多点综合对比图、误差对比柱状图、最差情况点对点对比图")
                    st.write(f"**验证点**: {', '.join([str(idx+1) for idx in plot_info['validation_indices']])}")
                    
                    # 显示统计信息
                    if 'statistics' in plot_info:
                        stats = plot_info['statistics']
                        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
                        with col_s1:
                            st.metric("平均相对误差", f"{stats['mean_relative_error']:.2f}%")
                        with col_s2:
                            st.metric("最大相对误差", f"{stats['max_relative_error']:.2f}%")
                        with col_s3:
                            st.metric("最小相对误差", f"{stats['min_relative_error']:.2f}%")
                        with col_s4:
                            st.metric("误差标准差", f"{stats['std_relative_error']:.2f}%")
                    
                    for j, fig in enumerate(plot_info['figures']):
                        st.pyplot(fig)
                
                elif plot_info['type'] == 'kfold_validation':
                    st.write("**包含图表**: K折交叉验证误差柱状图")
                    if 'errors' in plot_info:
                        errors = plot_info['errors']
                        col_k1, col_k2, col_k3 = st.columns(3)
                        with col_k1:
                            st.metric("平均误差", f"{np.mean(errors):.2e}")
                        with col_k2:
                            st.metric("最大误差", f"{np.max(errors):.2e}")
                        with col_k3:
                            st.metric("最小误差", f"{np.min(errors):.2e}")
                    st.pyplot(plot_info['figure'])
                
                else:
                    # 兼容旧的图表类型
                    if 'figures' in plot_info:
                        for j, fig in enumerate(plot_info['figures']):
                            st.pyplot(fig)
                    elif 'figure' in plot_info:
                        st.pyplot(plot_info['figure'])
        
        # 清除所有图表按钮
        if st.button("🗑️ 清除所有图表", type="secondary"):
            st.session_state.generated_plots = []
            st.success("✅ 所有图表已清除!")
            st.rerun()

# 底部信息
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 30px 0; background-color: #f8f9fa; border-radius: 10px; margin-top: 50px;'>
        <h4 style='margin: 0 0 10px 0; color: #666;'>🔧 模型降阶工具 v1.0</h4>
        <p style='margin: 0; color: #888;'>
            支持VTU和NPY文件格式 | 批量数据处理 | 多种降阶方法 | 3D可视化 | 预测精度分析
        </p>
    </div>
    """, 
    unsafe_allow_html=True
) 
