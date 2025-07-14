# BigLog 智慧运维模型应用指南

本文档提供关于BigLog模型在智慧运维领域的应用指南，包括日志事件匹配、非结构化日志分析、运维知识库接口开发和统一Web测试平台。

## 1. 项目结构

整合后的项目结构如下：

```
BigLog/
├── pretrained/                # 预训练模型文件目录
│   ├── config.json            # 模型配置文件
│   ├── pytorch_model.bin      # 模型权重文件
│   ├── tokenizer_config.json  # 分词器配置
│   ├── tokenizer.json         # 分词器参数
│   ├── special_tokens_map.json# 特殊标记映射
│   └── vocab.txt              # 词汇表
├── scripts/                   # 核心脚本目录
│   ├── unified_web_app.py     # 统一Web测试平台
│   ├── start_unified_app.py   # 统一平台启动脚本
│   ├── log_event_matching.py  # 日志事件匹配技术实现
│   ├── log_analysis_demo.py   # 非结构化日志分析实战
│   ├── ops_knowledge_api.py   # 运维知识库API实现
│   ├── start_api_server.py    # API服务启动脚本
│   ├── matplotlib_config.py   # Matplotlib中文字体配置
│   ├── install_fonts.py       # 中文字体安装工具
│   ├── knowledge_base.json    # 运维知识库数据
│   ├── sample_logs.txt        # 示例日志文件
│   ├── templates/             # Web模板目录
│   │   └── index.html         # 主页面模板
│   ├── static/                # 静态资源目录
│   │   └── style.css          # 样式文件
│   ├── uploads/               # 文件上传目录
│   ├── analysis_results/      # 分析结果目录
│   └── fonts/                 # 中文字体目录(可选)
└── BIGLOG.png                 # 项目Logo
```

## 2. 环境准备

### 2.1 安装依赖

```bash
pip install transformers==4.28.1 torch==2.0.0 scikit-learn==1.2.2 matplotlib==3.7.1 pandas==2.0.0 numpy==1.24.3 flask==2.3.2 seaborn==0.12.2
```

### 2.2 安装中文字体

为了正确显示图表中的中文字符，请安装中文字体：

```bash
cd scripts
python install_fonts.py
```

## 3. 统一Web测试平台

### 3.1 平台概述

BigLog统一测试平台是一个集成了所有测试功能的Web应用，提供简洁高效的界面进行日志分析、事件匹配和知识库管理。

**主要功能**：
- 📁 文件上传管理（按功能+日期自动重命名）
- 📊 日志聚类分析和异常检测
- 🔍 事件匹配和相似度计算
- 📚 知识库管理和语义搜索
- 📈 可视化结果展示

**启动方式**：

```bash
cd scripts
python start_unified_app.py
```

启动后浏览器会自动打开 http://localhost:5000

### 3.2 使用流程

1. **文件上传**：
   - 选择功能名称（如：web_server, database, application）
   - 上传日志文件（支持.txt, .log, .csv格式）
   - 系统自动按"功能+日期"格式重命名

2. **日志分析**：
   - 选择已上传的日志文件
   - 点击"开始分析"进行聚类和异常检测
   - 查看分析报告和可视化图表

3. **事件匹配**：
   - 输入查询日志内容
   - 选择候选日志文件
   - 获取相似度排序结果

4. **知识库管理**：
   - 添加知识条目（标题、内容、标签）
   - 语义搜索相关知识
   - 管理现有条目

5. **文件管理**：
   - 查看所有上传的文件
   - 下载分析结果
   - 预览生成的图表

## 4. 核心功能与测试流程

### 4.1 日志事件匹配

日志事件匹配技术用于识别相似日志并对其进行聚类，帮助运维人员快速定位相关事件。

**功能特点**：
- 基于BigLog模型的语义向量计算
- 日志相似度排序和匹配
- 聚类分析和可视化

**测试流程**：

```bash
cd scripts
python log_event_matching.py
```

**预期输出**：
- 日志相似度匹配结果
- 聚类分析结果
- 聚类可视化图（log_clusters.png）

### 4.2 非结构化日志分析

非结构化日志分析可以从大量原始日志中挖掘异常事件和模式，帮助提前发现系统问题。

**功能特点**：
- 日志预处理和特征提取
- 基于密度的聚类分析
- 异常日志检测
- 多维度可视化展示

**测试流程**：

```bash
cd scripts
python log_analysis_demo.py --log_file sample_logs.txt --output_dir ./analysis_results
```

**预期输出**：
- 聚类可视化（log_clusters_2d.png）
- 日志级别分布（log_level_distribution.png）
- 簇分析报告（cluster_analysis.txt）
- 异常日志列表（anomaly_logs.csv）

### 4.3 运维知识库API

运维知识库API提供基于语义的知识检索和管理功能，帮助运维人员快速获取解决方案。

**功能特点**：
- 基于BigLog的语义检索
- RESTful API设计
- 交互式API文档
- 知识条目管理

**启动服务**：

```bash
cd scripts
python start_api_server.py
```

**测试API**：

浏览器访问 http://localhost:5000/ 查看API文档，并使用以下命令测试API功能：

```bash
# 搜索知识库
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "数据库连接失败", "top_k": 3}'

# 添加知识条目
curl -X POST http://localhost:5000/api/add \
  -H "Content-Type: application/json" \
  -d '{
    "question": "如何排查网络连接问题?",
    "answer": "1. 检查网络配置\n2. 使用ping命令测试连通性\n3. 检查防火墙设置",
    "category": "network",
    "tags": ["网络", "连接", "故障排除"]
  }'
```

**API参数说明**：

```bash
# 自定义主机和端口
python start_api_server.py --host 127.0.0.1 --port 8080

# 指定模型路径
python start_api_server.py --model-path ../pretrained

# 启用调试模式
python start_api_server.py --debug
```

## 5. 故障排除

### 5.1 中文显示问题

如果图表中的中文显示为方框或乱码：

1. 运行字体安装工具，自动安装中文字体：
   ```bash
   cd scripts
   python install_fonts.py
   ```

2. 如果自动安装失败，可以手动将中文字体文件(如SimHei.ttf)复制到 `scripts/fonts/` 目录

3. 设置环境变量：`set PYTHONIOENCODING=utf-8` (Windows) 或 `export PYTHONIOENCODING=utf-8` (Linux/Mac)

### 5.2 日志事件匹配错误

如果运行`log_event_matching.py`时出现`NameError: name 'DBSCAN' is not defined`错误，说明缺少必要的导入。

解决方法：确保在脚本顶部正确导入了DBSCAN：
```python
from sklearn.cluster import KMeans, DBSCAN
```

### 5.3 编码问题

Windows环境下可能会遇到编码问题：

1. 运行 `chcp 65001` 设置控制台代码页为UTF-8
2. 确保所有文本文件使用UTF-8编码保存
3. 文件读写操作时显式指定编码：`open(file, 'r', encoding='utf-8')`

### 5.4 API服务无法启动

如果API服务无法启动：

1. 检查端口是否被占用，可以使用 `--port` 参数指定其他端口
2. 检查模型路径是否正确，确保所有模型文件存在
3. 检查依赖是否完整安装

## 6. 实际应用场景

### 6.1 服务器异常检测

通过非结构化日志分析功能，可以自动分析服务器日志，提前发现潜在问题：

```bash
python log_analysis_demo.py --log_file /var/log/system.log --output_dir ./system_analysis
```

### 6.2 智能故障诊断

结合日志事件匹配和知识库API，可以构建智能故障诊断系统：

1. 使用`log_event_matching.py`识别相似故障日志
2. 使用API搜索相关解决方案：`curl -X POST http://localhost:5000/api/search -H "Content-Type: application/json" -d '{"query": "已识别的故障模式"}'`
3. 基于匹配结果自动推荐解决方案

### 6.3 运维知识管理

利用知识库API构建企业运维知识管理平台：

1. 将积累的运维经验添加到知识库
2. 开发Web界面调用API进行知识检索和管理
3. 结合日志分析结果，自动丰富知识库内容

## 7. 定制与扩展

### 7.1 使用自己的日志数据

您可以使用自己的日志数据进行分析：

```bash
python log_analysis_demo.py --log_file 您的日志文件.log --output_dir 输出目录 --pattern "您的日志正则表达式"
```

### 7.2 扩展API功能

您可以通过以下方式扩展API功能：

1. 添加新的端点，如批量导入、删除等
2. 实现用户认证和权限控制
3. 添加缓存机制提高性能
4. 整合到更大的运维平台中

### 7.3 统一Web平台故障排除

#### 7.3.1 平台无法启动

如果统一Web平台无法启动：

1. 检查依赖是否完整安装：
   ```bash
   pip install flask torch transformers scikit-learn matplotlib numpy pandas
   ```

2. 检查必要文件是否存在：
   ```bash
   ls -la unified_web_app.py templates/index.html static/style.css
   ```

3. 检查端口是否被占用：
   ```bash
   # Windows
   netstat -ano | findstr :5000
   
   # Linux/Mac
   lsof -i :5000
   ```

#### 7.3.2 模型加载失败

如果模型加载失败：

1. 检查预训练模型路径：
   ```bash
   ls -la ../pretrained/
   ```

2. 确保模型文件完整：
   - config.json
   - pytorch_model.bin
   - tokenizer_config.json
   - tokenizer.json
   - special_tokens_map.json
   - vocab.txt

3. 如果模型文件不存在，应用将以有限功能模式运行

#### 7.3.3 文件上传失败

如果文件上传失败：

1. 检查上传目录权限：
   ```bash
   ls -la uploads/
   ```

2. 确保文件大小不超过16MB限制

3. 检查文件格式是否支持（.txt, .log, .csv）

#### 7.3.4 分析功能异常

如果分析功能异常：

1. 检查日志文件格式是否正确
2. 确保文件编码为UTF-8
3. 检查是否有足够的内存进行模型推理

#### 7.3.5 浏览器兼容性

统一Web平台支持以下浏览器：
- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

如果遇到显示问题，请尝试使用Chrome浏览器。 