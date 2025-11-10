@echo off
REM Streamlit部署前的初始化脚本 - Windows版本

REM 升级pip
echo 升级pip...
pip install --upgrade pip

REM 安装依赖
echo 安装项目依赖...
pip install -r requirements.txt --no-cache-dir --force-reinstall

REM 检查安装状态
echo 检查安装的包...
pip list | findstr /C:"langchain" /C:"streamlit" /C:"pydantic" /C:"faiss"

echo 依赖安装完成！