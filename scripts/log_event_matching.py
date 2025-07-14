#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
日志事件匹配技术
使用BigLog预训练模型进行相似日志识别和事件匹配
"""

import os
import sys
import json
import time
import random
import logging
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from collections import defaultdict
from transformers import AutoTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, DBSCAN
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE

# 导入中文字体配置模块
try:
    from matplotlib_config import configure_matplotlib_chinese
    configure_matplotlib_chinese()
except ImportError:
    logging.warning("未找到matplotlib_config模块，中文显示可能存在问题")

class LogEventMatcher:
    """使用BigLog模型进行日志事件匹配的类"""
    
    def __init__(self, model_path='../pretrained'):
        """
        初始化日志事件匹配器
        
        Args:
            model_path: BigLog预训练模型路径
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        self.model.eval()  # 设置为评估模式
        
    def get_log_embeddings(self, log_entries, max_length=150, batch_size=32):
        """
        获取日志条目的嵌入向量
        
        Args:
            log_entries: 日志条目列表
            max_length: 最大序列长度
            batch_size: 批处理大小
            
        Returns:
            日志嵌入向量
        """
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(log_entries), batch_size):
            batch = log_entries[i:i+batch_size]
            
            # 对日志进行分词
            encoded_input = self.tokenizer(
                batch, 
                padding='max_length',
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # 获取嵌入向量
            with torch.no_grad():
                output = self.model(**encoded_input)
                # 使用[CLS]令牌的表示作为整个日志的表示
                # 修改为在操作前先转到CPU，然后转为numpy
                embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
                
        # 确保数据类型一致
        if all_embeddings:
            return np.vstack(all_embeddings).astype(np.float32)
        else:
            return np.array([])
    
    def match_log_events(self, query_log, log_entries, top_k=5):
        """
        将查询日志与日志条目集合进行匹配
        
        Args:
            query_log: 查询日志条目
            log_entries: 要匹配的日志条目列表
            top_k: 返回最相似的条目数量
            
        Returns:
            最相似的日志条目索引和相似度分数
        """
        # 获取查询日志的嵌入向量
        query_embedding = self.get_log_embeddings([query_log])
        
        # 获取所有日志条目的嵌入向量
        log_embeddings = self.get_log_embeddings(log_entries)
        
        # 计算余弦相似度
        similarities = cosine_similarity(query_embedding, log_embeddings)[0]
        
        # 获取最相似的top_k个条目
        top_indices = similarities.argsort()[-top_k:][::-1]
        top_similarities = similarities[top_indices]
        
        return top_indices, top_similarities
    
    def cluster_log_events(self, log_entries, eps=0.5, min_samples=5):
        """
        将日志条目进行聚类
        
        Args:
            log_entries: 日志条目列表
            eps: DBSCAN的eps参数
            min_samples: DBSCAN的min_samples参数
            
        Returns:
            聚类标签
        """
        # 获取日志嵌入向量
        log_embeddings = self.get_log_embeddings(log_entries)
        
        # 使用DBSCAN进行聚类
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(log_embeddings)
        
        return clustering.labels_

def visualize_clusters(log_entries, labels, output_file='log_clusters.png'):
    """
    可视化日志聚类结果
    
    Args:
        log_entries: 日志条目列表
        labels: 聚类标签
        output_file: 输出文件路径
    """
    # 创建DataFrame
    df = pd.DataFrame({
        'log': log_entries,
        'cluster': labels
    })
    
    # 统计每个簇的大小
    cluster_sizes = df['cluster'].value_counts().sort_index()
    
    # 绘制条形图
    plt.figure(figsize=(12, 6))
    bars = plt.bar(cluster_sizes.index.astype(str), cluster_sizes.values)
    
    # 为-1簇（噪声）使用不同颜色
    if -1 in cluster_sizes.index:
        noise_idx = list(cluster_sizes.index).index(-1)
        bars[noise_idx].set_color('red')
    
    plt.xlabel('簇标签')
    plt.ylabel('日志条目数量')
    plt.title('日志聚类结果')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(output_file)
    plt.close()

def main():
    # 示例日志
    sample_logs = [
        "ERROR: Connection refused to database at 192.168.1.101:3306",
        "ERROR: Connection timed out after 30 seconds",
        "INFO: System startup completed in 12.5 seconds",
        "WARN: High CPU usage detected: 95%",
        "WARN: Memory usage exceeds 80% threshold",
        "ERROR: Connection refused to database at 192.168.1.105:3306",
        "INFO: User 'admin' logged in successfully",
        "INFO: Backup completed successfully",
        "ERROR: Failed to read configuration file",
        "WARN: Disk space is running low (15% free)",
    ]
    
    # 初始化日志事件匹配器
    matcher = LogEventMatcher()
    
    print("===== 日志事件匹配示例 =====")
    # 查询日志
    query = "ERROR: Connection refused to database"
    print(f"查询: {query}")
    
    # 匹配日志事件
    top_indices, top_similarities = matcher.match_log_events(query, sample_logs, top_k=3)
    
    # 打印结果
    print("\n最相似的日志:")
    for idx, sim in zip(top_indices, top_similarities):
        print(f"相似度: {sim:.4f} - {sample_logs[idx]}")
    
    print("\n===== 日志聚类示例 =====")
    # 对日志进行聚类
    cluster_labels = matcher.cluster_log_events(sample_logs, eps=0.6, min_samples=2)
    
    # 打印聚类结果
    for i, (log, label) in enumerate(zip(sample_logs, cluster_labels)):
        print(f"簇 {label}: {log}")
    
    # 可视化聚类结果
    visualize_clusters(sample_logs, cluster_labels)
    print("\n聚类结果已保存为 log_clusters.png")

if __name__ == "__main__":
    main() 