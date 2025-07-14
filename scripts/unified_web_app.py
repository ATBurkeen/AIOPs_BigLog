#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BigLog 统一测试平台
集成所有测试功能的Web应用
"""

import os
import json
import datetime
import shutil
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import rcParams
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import re
import logging
import uuid

# 配置日志系统
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'biglog_app.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置matplotlib中文字体
try:
    # 尝试使用系统中文字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
    font_found = False
    for font in chinese_fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            font_found = True
            break
        except:
            continue
    
    if not font_found:
        # 使用项目中的字体文件
        font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'simhei.ttf')
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    logger.warning(f"字体配置警告: {e}")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'biglog_unified_test_platform'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 512MB max file size

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('analysis_results', exist_ok=True)

# 全局变量存储数据
knowledge_base = []
log_embeddings = []
log_texts = []
analysis_results = {}

class BigLogAnalyzer:
    def __init__(self, model_path="../pretrained"):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
    
    def load_model(self):
        """加载预训练模型"""
        try:
            logger.info(f"正在加载模型: {self.model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModel.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            logger.info("模型加载成功!")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def extract_embeddings(self, texts, batch_size=16):
        """提取文本嵌入向量"""
        if not texts:
            return []
        
        # 添加文本长度控制，模型配置中max_position_embeddings为150
        logger.info(f"处理 {len(texts)} 条文本，最大长度限制为150")
        
        # 截断过长文本防止超出模型限制
        processed_texts = []
        for text in texts:
            if len(text) > 500:  # 超长文本预处理
                logger.warning(f"发现过长文本: {len(text)} 字符，将被截断")
                text = text[:500] + "..."
            processed_texts.append(text)
        
        embeddings = []
        
        for i in range(0, len(processed_texts), batch_size):
            batch_texts = processed_texts[i:i + batch_size]
            logger.info(f"处理批次 {i//batch_size + 1}/{(len(processed_texts) + batch_size - 1)//batch_size}")
            
            try:
                # 编码文本
                inputs = self.tokenizer(
                    batch_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=150,  # 显式设置为模型的max_position_embeddings
                    return_tensors="pt"
                )
                
                # 移动到设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 提取嵌入
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    # 使用[CLS]标记的输出作为句子嵌入
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"处理批次出错: {str(e)}")
                # 对出错的批次逐条处理
                logger.info("尝试对批次中的每条文本单独处理...")
                for text in batch_texts:
                    try:
                        single_input = self.tokenizer(
                            [text],
                            padding='max_length',
                            truncation=True,
                            max_length=150,
                            return_tensors="pt"
                        )
                        single_input = {k: v.to(self.device) for k, v in single_input.items()}
                        with torch.no_grad():
                            output = self.model(**single_input)
                            single_embedding = output.last_hidden_state[:, 0, :].cpu().numpy()
                            embeddings.extend(single_embedding)
                    except Exception as inner_e:
                        logger.error(f"单条文本处理失败: {str(inner_e)}")
                        # 用零向量代替失败的嵌入
                        zero_vector = np.zeros((1, 768))  # BERT模型的隐藏维度是768
                        embeddings.append(zero_vector[0])
        
        return np.array(embeddings)

# 初始化分析器
try:
    analyzer = BigLogAnalyzer()
    logger.info("BigLog分析器初始化成功")
except Exception as e:
    logger.error(f"BigLog分析器初始化失败: {e}")
    analyzer = None

def save_uploaded_file(file, function_name):
    """保存上传的文件，按功能+日期重命名"""
    if file and file.filename:
        # 获取当前日期
        current_date = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 生成新文件名
        file_extension = os.path.splitext(file.filename)[1]
        new_filename = f"{function_name}_{current_date}{file_extension}"
        
        # 保存文件
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        file.save(file_path)
        
        return file_path, new_filename
    return None, None

def load_knowledge_base():
    """加载知识库"""
    global knowledge_base
    
    # 使用正确的文件路径
    kb_file = os.path.join(os.path.dirname(__file__), 'knowledge_base.json')
    
    try:
        with open(kb_file, 'r', encoding='utf-8') as f:
            knowledge_base = json.load(f)
            logging.info(f"成功从 {kb_file} 加载知识库，包含 {len(knowledge_base)} 条目")
    except FileNotFoundError:
        knowledge_base = []
        # 创建空的知识库文件
        with open(kb_file, 'w', encoding='utf-8') as f:
            json.dump([], f)
        logging.info(f"在 {kb_file} 创建了新的知识库文件")
    except Exception as e:
        logging.error(f"加载知识库失败: {e}")
        knowledge_base = []
    
    # 确保返回值不为None
    if knowledge_base is None:
        knowledge_base = []
        logging.warning("知识库为None，已初始化为空列表")
    
    return knowledge_base

def save_knowledge_base(kb=None):
    """保存知识库"""
    global knowledge_base
    
    # 使用正确的文件路径
    kb_file = os.path.join(os.path.dirname(__file__), 'knowledge_base.json')
    
    try:
        # 如果提供了知识库参数，则更新全局知识库
        if kb is not None:
            knowledge_base = kb
        
        # 确保知识库不为None
        if knowledge_base is None:
            knowledge_base = []
            logging.warning("保存时知识库为None，已初始化为空列表")
            
        with open(kb_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge_base, f, ensure_ascii=False, indent=2)
        logging.info(f"知识库保存成功，保存了 {len(knowledge_base)} 条目到 {kb_file}")
    except Exception as e:
        logging.error(f"保存知识库失败: {e}")

# 初始化全局变量
knowledge_base = []

# 初始化知识库
knowledge_base = load_knowledge_base()

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """文件上传处理"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有选择文件'}), 400
        
        file = request.files['file']
        function_name = request.form.get('function_name', 'unknown')
        
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        file_path, new_filename = save_uploaded_file(file, function_name)
        
        if file_path:
            return jsonify({
                'success': True,
                'message': f'文件上传成功: {new_filename}',
                'filename': new_filename,
                'filepath': file_path
            })
        else:
            return jsonify({'error': '文件保存失败'}), 500
            
    except Exception as e:
        return jsonify({'error': f'上传失败: {str(e)}'}), 500

@app.route('/analyze_logs', methods=['POST'])
def analyze_logs():
    """日志分析"""
    try:
        plot_path = None
        if not analyzer:
            return jsonify({'error': '模型未加载'}), 500
        
        # 获取日志文件路径
        log_file = request.form.get('log_file')
        if not log_file:
            return jsonify({'error': '请选择日志文件'}), 400
        
        log_path = os.path.join(app.config['UPLOAD_FOLDER'], log_file)
        if not os.path.exists(log_path):
            return jsonify({'error': '日志文件不存在'}), 404
        
        logger.info(f"开始处理日志文件: {log_file}")
        
        # 读取日志文件
        log_texts = []
        file_extension = os.path.splitext(log_path)[1].lower()
        
        # 记录文件信息
        file_size = os.path.getsize(log_path) / 1024 / 1024  # MB
        logger.info(f"文件大小: {file_size:.2f} MB, 类型: {file_extension}")
        
        if file_extension == '.csv':
            # CSV格式日志处理
            try:
                # 尝试不同的编码方式读取CSV
                try:
                    df = pd.read_csv(log_path, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(log_path, encoding='gbk')
                    except UnicodeDecodeError:
                        df = pd.read_csv(log_path, encoding='latin1')
                
                # 为CSV日志创建格式化文本
                for _, row in df.iterrows():
                    # 创建简化的日志文本格式，避免过长
                    log_text = f"{row.get('level', 'INFO')}: "
                    
                    if 'module' in df.columns and 'action' in df.columns:
                        log_text += f"{row.get('module', '')}_{row.get('action', '')} "
                    
                    # 如果存在时间字段，添加时间信息
                    if 'createdTime' in df.columns:
                        log_text += f"[{row.get('createdTime', '')}] "
                    
                    # 简化data字段，避免过长
                    if 'data' in df.columns and pd.notna(row.get('data')) and row.get('data'):
                        try:
                            data = row.get('data')
                            # 如果是JSON字符串，只保留关键信息
                            if isinstance(data, str) and (data.startswith('{') or data.startswith('[')):
                                try:
                                    data_obj = json.loads(data)
                                    # 只保留关键字段作为摘要
                                    keys = list(data_obj.keys())[:3] if isinstance(data_obj, dict) else []
                                    data_summary = ", ".join([f"{k}:{data_obj[k]}" for k in keys])
                                    log_text += f"data: {{{data_summary}...}} "
                                except:
                                    # JSON解析失败，截断显示
                                    log_text += f"data: {data[:50]}... "
                            else:
                                # 非JSON字符串，直接截断
                                log_text += f"data: {str(data)[:50]} "
                        except:
                            pass
                    
                    # 添加设备和浏览器信息
                    if 'device' in df.columns:
                        log_text += f"device:{row.get('device', '')}"
                    if 'browser' in df.columns:
                        log_text += f" browser:{row.get('browser', '')}"
                    
                    log_texts.append(log_text)
            except Exception as e:
                return jsonify({'error': f'CSV日志解析失败: {str(e)}'}), 500
        else:
            # 纯文本日志处理
            with open(log_path, 'r', encoding='utf-8') as f:
                log_lines = f.readlines()
            
            # 预处理日志
            for line in log_lines:
                line = line.strip()
                if line:
                    log_texts.append(line)
        
        if not log_texts:
            return jsonify({'error': '日志文件为空'}), 400
        
        # 提取嵌入向量
        try:
            logger.info(f"开始提取嵌入向量，日志条数: {len(log_texts)}")
            embeddings = analyzer.extract_embeddings(log_texts)
            logger.info(f"嵌入向量提取完成，形状: {embeddings.shape}")
        except Exception as e:
            logger.error(f"嵌入向量提取失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'嵌入向量提取失败: {str(e)}'}), 500
        
        # 聚类分析
        n_clusters = min(5, len(log_texts))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # 异常检测（基于距离）
        distances = kmeans.transform(embeddings)
        min_distances = np.min(distances, axis=1)
        threshold = np.percentile(min_distances, 95)
        anomalies = min_distances > threshold
        
        # 生成可视化
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 聚类可视化
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(12, 8))
        
        # 聚类图
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=cluster_labels, cmap='viridis')
        plt.title('日志聚类分析')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.colorbar(scatter)
        
        # 异常检测图
        plt.subplot(2, 2, 2)
        normal_points = embeddings_2d[~anomalies]
        anomaly_points = embeddings_2d[anomalies]
        
        if len(normal_points) > 0:
            plt.scatter(normal_points[:, 0], normal_points[:, 1], c='blue', alpha=0.6, label='正常')
        if len(anomaly_points) > 0:
            plt.scatter(anomaly_points[:, 0], anomaly_points[:, 1], c='red', s=100, label='异常')
        
        plt.title('异常检测结果')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        
        # 距离分布图
        plt.subplot(2, 2, 3)
        plt.hist(min_distances, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(threshold, color='red', linestyle='--', label=f'阈值: {threshold:.3f}')
        plt.title('距离分布')
        plt.xlabel('到聚类中心的距离')
        plt.ylabel('频次')
        plt.legend()
        
        # 聚类大小分布
        plt.subplot(2, 2, 4)
        cluster_sizes = [int(np.sum(cluster_labels == i)) for i in range(n_clusters)]
        analysis_report = {
            'total_logs': int(len(log_texts)),
            'clusters': int(n_clusters),
            'anomalies': int(np.sum(anomalies)),
            'anomaly_rate': float(np.sum(anomalies) / len(log_texts) * 100),
            'cluster_sizes': cluster_sizes,
            'plot_path': plot_path
        }
        plt.bar(range(n_clusters), cluster_sizes, color='lightcoral')
        plt.title('聚类大小分布')
        plt.xlabel('聚类编号')
        plt.ylabel('日志数量')
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = f'analysis_results/log_analysis_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成分析报告
        analysis_report = {
            'total_logs': len(log_texts),
            'clusters': n_clusters,
            'anomalies': int(np.sum(anomalies)),
            'anomaly_rate': float(np.sum(anomalies) / len(log_texts) * 100),
            'cluster_sizes': cluster_sizes,
            'plot_path': plot_path
        }
        
        # 保存详细结果
        results_path = f'analysis_results/analysis_{timestamp}.json'
        detailed_results = {
            'log_texts': log_texts,
            'cluster_labels': cluster_labels.tolist(),
            'anomalies': anomalies.tolist(),
            'distances': min_distances.tolist(),
            'embeddings_2d': embeddings_2d.tolist()
        }
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)
        
        return jsonify({
            'success': True,
            'report': analysis_report,
            'plot_path': plot_path,
            'results_path': results_path
        })
        
    except Exception as e:
        return jsonify({'error': f'分析失败: {str(e)}'}), 500

@app.route('/event_matching', methods=['POST'])
def event_matching():
    """事件匹配"""
    try:
        if not analyzer:
            return jsonify({'error': '模型未加载'}), 500
        
        # 获取查询日志
        query_log = request.form.get('query_log')
        if not query_log:
            return jsonify({'error': '请输入查询日志'}), 400
        
        # 获取候选日志文件
        candidate_file = request.form.get('candidate_file')
        if not candidate_file:
            return jsonify({'error': '请选择候选日志文件'}), 400
        
        candidate_path = os.path.join(app.config['UPLOAD_FOLDER'], candidate_file)
        if not os.path.exists(candidate_path):
            return jsonify({'error': '候选日志文件不存在'}), 404
        
        # 读取候选日志
        candidate_logs = []
        raw_logs = []  # 保存原始日志数据，用于知识库
        file_extension = os.path.splitext(candidate_path)[1].lower()
        
        if file_extension == '.csv':
            # CSV格式日志处理
            try:
                # 尝试不同的编码方式读取CSV
                try:
                    df = pd.read_csv(candidate_path, encoding='utf-8')
                except UnicodeDecodeError:
                    try:
                        df = pd.read_csv(candidate_path, encoding='gbk')
                    except UnicodeDecodeError:
                        df = pd.read_csv(candidate_path, encoding='latin1')
                
                # 为CSV日志创建格式化文本
                for _, row in df.iterrows():
                    # 保存原始日志数据
                    raw_logs.append(row.to_dict())
                    
                    # 创建简化的日志文本格式
                    log_text = f"{row.get('level', 'INFO')}: "
                    
                    if 'module' in df.columns and 'action' in df.columns:
                        log_text += f"{row.get('module', '')}_{row.get('action', '')} "
                    
                    if 'createdTime' in df.columns:
                        log_text += f"[{row.get('createdTime', '')}] "
                    
                    # 简化data字段
                    if 'data' in df.columns and pd.notna(row.get('data')) and row.get('data'):
                        data = row.get('data')
                        # 截断长数据
                        log_text += f"data: {str(data)[:50]}... "
                    
                    if 'device' in df.columns:
                        log_text += f"device:{row.get('device', '')}"
                        
                    candidate_logs.append(log_text)
            except Exception as e:
                return jsonify({'error': f'CSV日志解析失败: {str(e)}'}), 500
        else:
            # 纯文本日志处理
            with open(candidate_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                candidate_logs = lines
                raw_logs = lines  # 对于文本日志，原始数据就是文本行
        
        if not candidate_logs:
            return jsonify({'error': '候选日志文件为空'}), 400
        
        # 提取嵌入向量
        all_texts = [query_log] + candidate_logs
        embeddings = analyzer.extract_embeddings(all_texts)
        
        query_embedding = embeddings[0:1]
        candidate_embeddings = embeddings[1:]
        
        # 计算相似度
        similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]
        
        # 排序结果
        results = []
        for i, (log, sim, raw_log) in enumerate(zip(candidate_logs, similarities, raw_logs)):
            results.append({
                'rank': i + 1,
                'log': log,
                'raw_log': raw_log,
                'similarity': float(sim)
            })
        
        # 按相似度排序
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # 生成可视化
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plt.figure(figsize=(12, 6))
        
        # 相似度分布
        plt.subplot(1, 2, 1)
        similarities_sorted = sorted(similarities, reverse=True)
        plt.plot(range(len(similarities_sorted)), similarities_sorted, 'b-', linewidth=2)
        plt.scatter(range(len(similarities_sorted)), similarities_sorted, c='red', s=50)
        plt.title('相似度分布')
        plt.xlabel('候选日志排名')
        plt.ylabel('相似度分数')
        plt.grid(True, alpha=0.3)
        
        # Top-10相似度柱状图
        plt.subplot(1, 2, 2)
        top_10 = results[:10]
        ranks = [r['rank'] for r in top_10]
        sims = [r['similarity'] for r in top_10]
        plt.bar(range(len(top_10)), sims, color='lightblue', edgecolor='black')
        plt.title('Top-10 相似度结果')
        plt.xlabel('排名')
        plt.ylabel('相似度分数')
        plt.xticks(range(len(top_10)), ranks)
        
        plt.tight_layout()
        
        # 保存图片
        plot_path = f'analysis_results/event_matching_{timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 保存匹配结果到知识库
        top_results = results[:5]  # 取前5个最相似的结果
        knowledge_entry = create_knowledge_entry_from_matching(query_log, top_results, plot_path, timestamp)
        
        # 搜索知识库中的相关条目
        kb_results = []
        kb = load_knowledge_base()
        
        if kb and len(kb) > 0:
            # 提取查询日志中的关键词
            query_keywords = extract_keywords(query_log)
            logger.info(f"从查询日志中提取的关键词: {query_keywords}")
            
            # 基于标签匹配
            for entry in kb:
                # 获取条目标签
                entry_tags = entry.get('tags', [])
                if not entry_tags:
                    continue
                
                # 计算标签匹配分数
                match_score = 0
                matched_tags = []
                
                for keyword in query_keywords:
                    for tag in entry_tags:
                        # 如果关键词包含在标签中或标签包含在关键词中
                        if keyword.lower() in tag.lower() or tag.lower() in keyword.lower():
                            match_score += 1
                            if tag not in matched_tags:
                                matched_tags.append(tag)
                
                # 如果有匹配的标签
                if match_score > 0:
                    # 计算匹配分数 (0-1之间)
                    similarity = min(match_score / max(len(query_keywords), 1), 1.0)
                    
                    kb_results.append({
                        'entry': entry,
                        'similarity': float(similarity),
                        'matched_tags': matched_tags
                    })
            
            # 按相似度排序
            kb_results.sort(key=lambda x: x['similarity'], reverse=True)
            kb_results = kb_results[:5]  # 只返回前5个最相似的结果
            
            logger.info(f"基于标签匹配找到 {len(kb_results)} 个知识库条目")
        
        return jsonify({
            'success': True,
            'query_log': query_log,
            'results': results[:20],  # 返回前20个结果
            'plot_path': plot_path,
            'knowledge_entry': knowledge_entry,
            'kb_results': kb_results  # 返回知识库匹配结果
        })
        
    except Exception as e:
        return jsonify({'error': f'事件匹配失败: {str(e)}'}), 500

def create_knowledge_entry_from_matching(query_log, top_results, plot_path, timestamp):
    """
    从事件匹配结果创建知识库条目
    
    Args:
        query_log: 查询日志
        top_results: 匹配结果（前N个）
        plot_path: 可视化图表路径
        timestamp: 时间戳
        
    Returns:
        创建的知识库条目
    """
    try:
        # 加载知识库
        kb = load_knowledge_base()
        
        # 确保知识库不为None
        if kb is None:
            kb = []
            logging.warning("知识库为None，已初始化为空列表")
        
        # 生成唯一ID
        entry_id = str(uuid.uuid4())
        
        # 提取标签
        tags = []
        
        # 1. 添加日志级别标签
        if "ERROR" in query_log.upper():
            tags.append("错误")
        if "WARN" in query_log.upper() or "WARNING" in query_log.upper():
            tags.append("警告")
        if "INFO" in query_log.upper():
            tags.append("信息")
        if "DEBUG" in query_log.upper():
            tags.append("调试")
            
        # 2. 提取关键词作为标签
        keywords = extract_keywords(query_log)
        
        # 3. 从匹配结果中提取可能的模块名或错误类型
        for result in top_results[:2]:  # 只考虑前两个最相似的结果
            log_text = result['log']
            # 尝试提取模块名
            module_match = re.search(r'(\w+)_(\w+)', log_text)
            if module_match and module_match.group(1) not in tags:
                tags.append(module_match.group(1))  # 添加模块名
            
            # 尝试提取错误类型
            error_match = re.search(r'([A-Z][a-zA-Z]+Error|Exception|Failure)', log_text)
            if error_match and error_match.group(1) not in tags:
                tags.append(error_match.group(1))
        
        # 4. 添加关键词作为标签
        for keyword in keywords:
            if keyword not in tags and len(keyword) >= 4:
                tags.append(keyword)
        
        # 5. 限制标签数量并去重
        tags = list(set(tags))[:8]  # 最多8个标签
        
        # 构建内容
        content = f"查询日志: {query_log}\n\n匹配结果:\n"
        for i, result in enumerate(top_results):
            similarity_percent = round(result['similarity'] * 100, 2)
            content += f"{i+1}. 相似度: {similarity_percent}% - {result['log']}\n"
        
        content += f"\n可视化图表: {plot_path}"
        
        # 创建知识库条目
        entry = {
            "id": entry_id,
            "title": f"日志事件匹配: {query_log[:50]}{'...' if len(query_log) > 50 else ''}",
            "content": content,
            "tags": tags,
            "timestamp": datetime.datetime.now().isoformat(),
            "source": "event_matching",
            "related_image": plot_path
        }
        
        # 添加到知识库
        kb.append(entry)
        
        # 保存知识库
        save_knowledge_base(kb)
        
        return entry
    except Exception as e:
        logging.error(f"创建知识库条目失败: {str(e)}")
        return None

@app.route('/knowledge_base', methods=['GET', 'POST'])
def knowledge_base_api():
    """知识库管理"""
    global knowledge_base
    
    # 确保知识库已初始化
    if knowledge_base is None:
        knowledge_base = load_knowledge_base()
    
    if request.method == 'GET':
        try:
            return jsonify({
                'success': True,
                'entries': knowledge_base,
                'count': len(knowledge_base)
            })
        except Exception as e:
            logging.error(f"获取知识库失败: {str(e)}")
            return jsonify({
                'success': False,
                'error': f"获取知识库失败: {str(e)}",
                'entries': [],
                'count': 0
            })
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            action = data.get('action')
            
            if action == 'add':
                # 使用UUID作为ID，与事件匹配创建的条目保持一致
                entry = {
                    'id': str(uuid.uuid4()),
                    'title': data.get('title', ''),
                    'content': data.get('content', ''),
                    'tags': data.get('tags', []),
                    'timestamp': datetime.datetime.now().isoformat()
                }
                knowledge_base.append(entry)
                save_knowledge_base(knowledge_base)
                
                return jsonify({
                    'success': True,
                    'message': '条目添加成功',
                    'entry': entry
                })
            
            elif action == 'search':
                query = data.get('query', '')
                if not query:
                    return jsonify({'error': '请输入搜索查询'}), 400
                
                if not analyzer:
                    return jsonify({'error': '模型未加载'}), 500
                
                # 语义搜索
                query_embedding = analyzer.extract_embeddings([query])
                
                # 提取所有条目的嵌入
                all_contents = [entry['content'] for entry in knowledge_base]
                if not all_contents:
                    return jsonify({
                        'success': True,
                        'results': [],
                        'count': 0
                    })
                
                content_embeddings = analyzer.extract_embeddings(all_contents)
                
                # 计算相似度
                similarities = cosine_similarity(query_embedding, content_embeddings)[0]
                
                # 排序结果
                results = []
                for i, (entry, sim) in enumerate(zip(knowledge_base, similarities)):
                    results.append({
                        'entry': entry,
                        'similarity': float(sim)
                    })
                
                results.sort(key=lambda x: x['similarity'], reverse=True)
                
                return jsonify({
                    'success': True,
                    'query': query,
                    'results': results[:10],  # 返回前10个结果
                    'count': len(results)
                })
            
            elif action == 'delete':
                entry_id = data.get('id')
                if entry_id is None:
                    return jsonify({'error': '请提供条目ID'}), 400
                
                # 记录删除前的条目数
                before_count = len(knowledge_base)
                
                # 将ID转为字符串进行比较，确保类型一致
                entry_id_str = str(entry_id)
                knowledge_base = [entry for entry in knowledge_base if str(entry.get('id', '')) != entry_id_str]
                
                # 检查是否真的删除了条目
                after_count = len(knowledge_base)
                if before_count == after_count:
                    logging.warning(f"未找到ID为 {entry_id} 的条目进行删除")
                    return jsonify({
                        'success': False,
                        'error': f"未找到ID为 {entry_id} 的条目"
                    }), 404
                
                # 保存知识库
                save_knowledge_base(knowledge_base)
                
                logging.info(f"成功删除ID为 {entry_id} 的知识库条目")
                return jsonify({
                    'success': True,
                    'message': '条目删除成功'
                })
            
            else:
                return jsonify({'error': '不支持的操作'}), 400
                
        except Exception as e:
            return jsonify({'error': f'操作失败: {str(e)}'}), 500

@app.route('/files')
def list_files():
    """列出上传的文件"""
    try:
        files = []
        upload_dir = app.config['UPLOAD_FOLDER']
        
        if os.path.exists(upload_dir):
            for filename in os.listdir(upload_dir):
                file_path = os.path.join(upload_dir, filename)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    files.append({
                        'name': filename,
                        'size': stat.st_size,
                        'modified': datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        return jsonify({
            'success': True,
            'files': files
        })
        
    except Exception as e:
        return jsonify({'error': f'获取文件列表失败: {str(e)}'}), 500

@app.route('/results')
def list_results():
    """列出分析结果"""
    try:
        results = []
        results_dir = 'analysis_results'
        
        if os.path.exists(results_dir):
            for filename in os.listdir(results_dir):
                if filename.endswith(('.png', '.json')):
                    file_path = os.path.join(results_dir, filename)
                    stat = os.stat(file_path)
                    results.append({
                        'name': filename,
                        'type': 'image' if filename.endswith('.png') else 'data',
                        'size': stat.st_size,
                        'modified': datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': f'获取结果列表失败: {str(e)}'}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    """下载文件"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': f'下载失败: {str(e)}'}), 500

@app.route('/view_result/<path:filename>')
def view_result(filename):
    """查看分析结果"""
    try:
        file_path = os.path.join('analysis_results', filename)
        if os.path.exists(file_path):
            if filename.endswith('.png'):
                return send_file(file_path, mimetype='image/png')
            elif filename.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return jsonify(data)
            else:
                return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': f'查看失败: {str(e)}'}), 500

@app.route('/knowledge_base/<string:entry_id>', methods=['GET'])
def get_knowledge_entry(entry_id):
    """获取指定ID的知识库条目"""
    global knowledge_base
    
    # 确保知识库已初始化
    if knowledge_base is None:
        knowledge_base = load_knowledge_base()
    
    try:
        # 查找指定ID的条目
        entry = next((e for e in knowledge_base if str(e.get('id')) == entry_id), None)
        
        if entry:
            return jsonify({
                'success': True,
                'entry': entry
            })
        else:
            return jsonify({
                'success': False,
                'error': f'未找到ID为 {entry_id} 的知识库条目'
            }), 404
    except Exception as e:
        logging.error(f"获取知识库条目失败: {str(e)}")
        return jsonify({
            'success': False,
            'error': f"获取知识库条目失败: {str(e)}"
        }), 500

def extract_keywords(text):
    """
    从文本中提取关键词
    
    Args:
        text: 要提取关键词的文本
        
    Returns:
        关键词列表
    """
    # 分词
    words = text.split()
    
    # 过滤掉常见的日志级别和短词
    stopwords = ['info', 'error', 'warn', 'warning', 'debug', 'trace', 'fatal', 'notice', 
                'the', 'and', 'is', 'in', 'at', 'to', 'for', 'with', 'by', 'on', 'of']
    
    keywords = []
    for word in words:
        # 清理标点符号
        word = word.strip('.,;:()[]{}"\'-').lower()
        
        # 过滤短词和停用词
        if len(word) >= 3 and word.lower() not in stopwords:
            keywords.append(word)
    
    # 添加一些特殊处理：提取可能的错误代码或标识符
    error_codes = re.findall(r'[A-Z]+\d+', text)  # 例如 E404, HTTP404
    error_codes += re.findall(r'[A-Z][A-Z0-9_]+', text)  # 例如 ERROR_CONNECTION
    
    # 合并关键词
    keywords.extend(error_codes)
    
    # 去重
    return list(set(keywords))

if __name__ == '__main__':
    logger.info("启动BigLog统一测试平台...")
    logger.info("访问地址: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 