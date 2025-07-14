# BigLog API 使用指南

这个文档提供了如何使用BigLog API和相关实验脚本的指导。

## 1. 环境准备

确保您已经安装了所有必要的依赖：

```bash
pip install torch transformers scikit-learn matplotlib pandas numpy flask seaborn
```

## 2. 项目结构

- `ops_knowledge_api.py`: 运维知识库API实现
- `log_event_matching.py`: 日志事件匹配技术实现
- `log_analysis_demo.py`: 非结构化日志分析实战
- `start_api_server.py`: API服务启动脚本
- `matplotlib_config.py`: Matplotlib中文字体配置
- `knowledge_base.json`: 运维知识库文件
- `fonts/`: 可选的中文字体目录(如果需要)

## 3. 快速开始

### 启动运维知识库API服务

```bash
# 使用默认设置启动API
python start_api_server.py

# 指定主机和端口启动API
python start_api_server.py --host 127.0.0.1 --port 8080

# 指定模型路径启动API
python start_api_server.py --model-path ../pretrained
```

启动后，可以通过浏览器访问 http://localhost:5000/ 查看API文档。

### 运行日志事件匹配实验

```bash
python log_event_matching.py
```

### 运行日志分析演示

```bash
python log_analysis_demo.py
```

## 4. API使用示例

### 搜索知识库

```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "数据库连接失败", "top_k": 3}'
```

### 添加知识条目

```bash
curl -X POST http://localhost:5000/api/add \
  -H "Content-Type: application/json" \
  -d '{
    "question": "如何排查网络连接问题?",
    "answer": "1. 检查网络配置\n2. 使用ping命令测试连通性\n3. 检查防火墙设置",
    "category": "network",
    "tags": ["网络", "连接", "故障排除"]
  }'
```

## 5. 常见问题解决

### 中文显示问题

如果图表中的中文显示为方框或乱码，解决方案有：

1. 将中文字体文件(如SimHei.ttf)复制到`scripts/fonts/`目录
2. 修改`matplotlib_config.py`中的字体配置
3. 将环境变量`PYTHONIOENCODING`设置为`utf-8`

### 编码问题

如果遇到编码相关错误，请确保：

1. 所有Python文件使用UTF-8编码
2. 文本文件读写时指定UTF-8编码
3. Windows系统下设置控制台代码页：`chcp 65001`

## 6. 联系与支持

如有问题，请提交Issue或联系开发团队。 