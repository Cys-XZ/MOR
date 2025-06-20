#!/usr/bin/env python
"""
位移数据处理工具启动脚本
运行此脚本将启动Streamlit应用
"""

import subprocess
import sys
import os

def main():
    """启动Streamlit应用"""
    try:
        print("🚀 正在启动位移数据处理工具...")
        print("📱 应用将在浏览器中自动打开")
        print("🌐 默认地址: http://localhost:8501")
        print("⏹️  按 Ctrl+C 停止应用")
        print("-" * 50)
        
        # 启动Streamlit应用
        subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
        
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"❌ 启动应用时出错: {e}")
        print("💡 请确保已安装所有依赖包: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 