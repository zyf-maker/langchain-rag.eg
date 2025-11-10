@echo off

REM 升级pip到最新版本
echo 升级pip...
pip install --upgrade pip

REM 安装项目依赖
echo 安装项目依赖...
pip install -r requirements.txt --no-cache-dir --force-reinstall

REM 检查关键包是否安装成功
echo 检查关键包安装状态...
pip show langchain >nul 2>&1 && echo ✓ langchain 安装成功 || echo ✗ langchain 安装失败
pip show langchain_community >nul 2>&1 && echo ✓ langchain_community 安装成功 || echo ✗ langchain_community 安装失败
pip show chromadb >nul 2>&1 && echo ✓ chromadb 安装成功 || echo ✗ chromadb 安装失败
pip show streamlit >nul 2>&1 && echo ✓ streamlit 安装成功 || echo ✗ streamlit 安装失败
pip show pydantic >nul 2>&1 && echo ✓ pydantic 安装成功 || echo ✗ pydantic 安装失败

echo 依赖安装完成！