# Streamlit Cloud 部署指南

## 🚀 快速部署到Streamlit Cloud

### 1. 准备Git仓库
```bash
git init
git add .
git commit -m "Initial commit: ETH Monitor System"
git push origin main
```

### 2. 部署到Streamlit Cloud
1. 访问 [share.streamlit.io](https://share.streamlit.io)
2. 使用GitHub账号登录
3. 点击 "New app"
4. 选择你的仓库
5. 主文件路径填写: `增强版ETH监控系统.py`
6. 点击 "Deploy!"

### 3. 常见问题解决

#### 依赖安装失败
- 确保 `requirements.txt` 包含所有必要依赖
- 检查依赖版本兼容性

#### 内存不足
- Streamlit Cloud免费版有内存限制
- 考虑减少模型复杂度或数据量

#### 模块导入错误
确保以下依赖在requirements.txt中：
```
streamlit>=1.28.0
streamlit-autorefresh>=1.0.1
xgboost>=1.7.0
lightgbm>=3.3.0
scikit-learn>=1.3.0
```

### 4. 监控应用状态
- 在Streamlit Cloud控制台查看日志
- 监控应用健康状态
- 设置重启策略

### 5. 优化建议
- 使用@st.cache_data缓存数据
- 定期清理会话状态
- 监控内存使用情况 