# RAG应用部署指南

由于Git推送遇到网络问题，本指南提供了通过GitHub网页界面手动更新文件并完成Streamlit Cloud部署的步骤。

## 1. 通过GitHub网页更新requirements.txt

1. 访问你的GitHub仓库：https://github.com/zyf-maker/langchain-rag.eg
2. 在仓库主页找到并点击 `requirements.txt` 文件
3. 点击右上角的编辑按钮（铅笔图标）
4. 复制本地 `requirements.txt` 的内容并粘贴到编辑框中
5. 滚动到页面底部，添加提交信息（如："更新依赖版本以解决部署问题"）
6. 点击 "Commit changes" 按钮完成更新

## 2. Streamlit Cloud部署步骤

1. 访问 Streamlit Cloud：https://share.streamlit.io/
2. 登录你的账户并找到你的应用
3. 点击 "Manage app" 按钮
4. 在应用管理页面，点击 "Redeploy" 按钮开始重新部署
5. 查看部署日志，确认依赖安装成功

## 3. 本地测试部署（可选）

如果你想在本地测试依赖安装：

### Windows用户
```bash
# 运行批处理脚本
setup.bat
```

### Linux/Mac用户
```bash
# 赋予执行权限并运行脚本
chmod +x setup.sh
./setup.sh
```

## 4. 故障排除

如果部署仍然失败，请检查：

1. 确保所有环境变量都已在Streamlit Cloud中正确配置
2. 如果应用使用了特定的API密钥，请确保它们已设置
3. 查看完整的部署日志以获取详细错误信息

## 5. 回退方案

如果Streamlit Cloud持续遇到问题，可以考虑：

1. 使用Docker容器化部署
2. 部署到其他平台如Heroku、Render或AWS
3. 简化应用功能，减少依赖复杂度