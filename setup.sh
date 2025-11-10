#!/bin/bash
# Streamlit部署前的初始化脚本

# 升级pip
echo "升级pip..."
pip install --upgrade pip

# 安装依赖
echo "安装项目依赖..."
pip install -r requirements.txt --no-cache-dir --force-reinstall

# 检查关键包是否安装成功
echo "检查关键包安装状态..."
pip show langchain 2>&1 >/dev/null && echo "✓ langchain 安装成功" || echo "✗ langchain 安装失败"
pip show langchain_community 2>&1 >/dev/null && echo "✓ langchain_community 安装成功" || echo "✗ langchain_community 安装失败"
pip show chromadb 2>&1 >/dev/null && echo "✓ chromadb 安装成功" || echo "✗ chromadb 安装失败"
pip show streamlit 2>&1 >/dev/null && echo "✓ streamlit 安装成功" || echo "✗ streamlit 安装失败"
pip show pydantic 2>&1 >/dev/null && echo "✓ pydantic 安装成功" || echo "✗ pydantic 安装失败"
echo "依赖安装完成！"