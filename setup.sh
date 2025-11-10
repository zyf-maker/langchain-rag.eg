#!/bin/bash
# Streamlit部署前的初始化脚本

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo "安装项目依赖..."
pip install -r requirements.txt --no-cache-dir --force-reinstall

# 检查安装状态
echo "检查安装的包..."
pip list | grep -E "langchain|streamlit|pydantic|faiss"

echo "依赖安装完成！"