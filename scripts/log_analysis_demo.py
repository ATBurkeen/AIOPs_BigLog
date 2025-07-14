#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
非结构化日志分析实战
使用BigLog预训练模型进行日志聚类、异常检测及可视化
"""

import os
import sys
import json
import time
import logging
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from collections import Counter, defaultdict
from transformers import AutoTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import re
import seaborn as sns
import argparse

# 导入中文字体配置模块
try:
    from matplotlib_config import configure_matplotlib_chinese
    configure_matplotlib_chinese()
except ImportError:
    logging.warning("未找到matplotlib_config模块，中文显示可能存在问题")

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LogAnalyzer:
    """非结构化日志分析器类"""
    
    def __init__(self, model_path='../pretrained'):
        """
        初始化日志分析器
        
        Args:
            model_path: BigLog预训练模型路径
        """
        logger.info(f"正在加载BigLog模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        self.model.eval()
    
    def preprocess_logs(self, log_file, patterns=None):
        """
        预处理日志文件
        
        Args:
            log_file: 日志文件路径
            patterns: 正则表达式模式字典，用于提取日志元数据
            
        Returns:
            处理后的日志DataFrame
        """
        logger.info(f"正在预处理日志文件: {log_file}")
        
        # 默认解析模式
        if patterns is None:
            patterns = {
                'timestamp': r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
                'level': r'\[(INFO|ERROR|WARN|DEBUG)\]',
                'component': r'\[([^\]]+)\]',
                'content': r'(?:\[[^\]]+\]\s*)+(.+)'
            }
        
        logs = []
        with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                log_entry = {'raw': line}
                
                # 提取元数据
                for key, pattern in patterns.items():
                    match = re.search(pattern, line)
                    if match:
                        log_entry[key] = match.group(1).strip()
                    else:
                        log_entry[key] = None
                
                logs.append(log_entry)
        
        df = pd.DataFrame(logs)
        logger.info(f"预处理完成，共处理 {len(df)} 条日志")
        return df
    
    def extract_log_embeddings(self, log_texts, batch_size=32, max_length=150):
        """
        提取日志的嵌入向量表示
        
        Args:
            log_texts: 日志文本列表
            batch_size: 批处理大小
            max_length: 最大序列长度
            
        Returns:
            日志嵌入向量
        """
        logger.info(f"正在提取 {len(log_texts)} 条日志的嵌入向量...")
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(log_texts), batch_size):
            batch = log_texts[i:i+batch_size]
            
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
                embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)
                
        embeddings = np.vstack(all_embeddings)
        logger.info(f"嵌入向量提取完成，形状: {embeddings.shape}")
        return embeddings
    
    def cluster_logs(self, embeddings, eps=0.5, min_samples=5):
        """
        对日志嵌入向量进行聚类
        
        Args:
            embeddings: 日志嵌入向量
            eps: DBSCAN的eps参数
            min_samples: DBSCAN的min_samples参数
            
        Returns:
            聚类标签
        """
        logger.info(f"正在对日志进行聚类，参数: eps={eps}, min_samples={min_samples}")
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(embeddings)
        labels = clustering.labels_
        
        # 统计聚类结果
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        logger.info(f"聚类完成，发现 {n_clusters} 个簇，{n_noise} 条噪声日志")
        return labels
    
    def visualize_clusters(self, embeddings, labels, output_file='log_clusters_2d.png'):
        """
        使用t-SNE将高维嵌入向量降至2维并可视化聚类结果
        
        Args:
            embeddings: 日志嵌入向量
            labels: 聚类标签
            output_file: 输出文件路径
        """
        logger.info("正在使用t-SNE进行降维...")
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # 绘制散点图
        plt.figure(figsize=(12, 10))
        
        # 创建颜色映射
        unique_labels = set(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        # 绘制每个簇的点
        for i, (label, color) in enumerate(zip(unique_labels, colors)):
            if label == -1:
                # 噪声点用黑色表示
                color = [0, 0, 0, 1]
            
            mask = labels == label
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[color],
                label=f'Cluster {label}' if label != -1 else 'Noise',
                alpha=0.7
            )
        
        plt.title('t-SNE日志聚类可视化')
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        logger.info(f"聚类可视化已保存到: {output_file}")
    
    def analyze_cluster_patterns(self, df, labels, output_dir='.'):
        """
        分析每个簇的模式和特征
        
        Args:
            df: 日志DataFrame
            labels: 聚类标签
            output_dir: 输出目录
            
        Returns:
            簇分析结果
        """
        logger.info("正在分析簇模式和特征...")
        
        # 将标签添加到DataFrame
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = labels
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 分析每个簇
        cluster_analysis = {}
        unique_labels = sorted(set(labels))
        
        for label in unique_labels:
            cluster_logs = df_with_clusters[df_with_clusters['cluster'] == label]
            
            # 跳过噪声簇的详细分析
            if label == -1:
                cluster_analysis[label] = {
                    'size': len(cluster_logs),
                    'type': 'noise'
                }
                continue
            
            # 统计日志级别分布
            level_counts = cluster_logs['level'].value_counts().to_dict()
            
            # 提取最常见的组件
            component_counts = cluster_logs['component'].value_counts().head(5).to_dict()
            
            # 提取簇中最有代表性的日志内容
            # 选择簇中最中心的5个样本作为代表
            if 'content' in cluster_logs.columns:
                representative_logs = cluster_logs['content'].sample(min(5, len(cluster_logs))).tolist()
            else:
                representative_logs = cluster_logs['raw'].sample(min(5, len(cluster_logs))).tolist()
            
            # 计算簇内日志的平均长度
            avg_length = cluster_logs['raw'].str.len().mean()
            
            cluster_analysis[label] = {
                'size': len(cluster_logs),
                'level_distribution': level_counts,
                'top_components': component_counts,
                'representative_logs': representative_logs,
                'avg_log_length': avg_length
            }
            
            # 保存每个簇的日志到单独的文件
            cluster_file = os.path.join(output_dir, f'cluster_{label}_logs.csv')
            cluster_logs.to_csv(cluster_file, index=False)
            logger.info(f"簇 {label} 的日志已保存到: {cluster_file}")
        
        # 生成簇分析报告
        with open(os.path.join(output_dir, 'cluster_analysis.txt'), 'w', encoding='utf-8') as f:
            f.write(f"日志聚类分析报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总日志数: {len(df)}\n")
            f.write(f"簇数量: {len(unique_labels) - (1 if -1 in unique_labels else 0)}\n\n")
            
            for label in sorted(cluster_analysis.keys()):
                if label == -1:
                    f.write(f"噪声日志 (簇 -1):\n")
                    f.write(f"  数量: {cluster_analysis[label]['size']}\n\n")
                    continue
                
                analysis = cluster_analysis[label]
                f.write(f"簇 {label}:\n")
                f.write(f"  数量: {analysis['size']}\n")
                
                if 'level_distribution' in analysis:
                    f.write(f"  日志级别分布:\n")
                    for level, count in analysis['level_distribution'].items():
                        f.write(f"    {level}: {count}\n")
                
                if 'top_components' in analysis:
                    f.write(f"  主要组件:\n")
                    for component, count in analysis['top_components'].items():
                        if component is not None:
                            f.write(f"    {component}: {count}\n")
                
                if 'representative_logs' in analysis:
                    f.write(f"  代表性日志样本:\n")
                    for i, log in enumerate(analysis['representative_logs']):
                        if log is not None:
                            f.write(f"    {i+1}. {log[:100]}...\n")
                
                f.write("\n")
        
        logger.info(f"簇分析报告已保存到: {os.path.join(output_dir, 'cluster_analysis.txt')}")
        return cluster_analysis
    
    def detect_anomalies(self, df, labels):
        """
        基于聚类结果检测异常日志
        
        Args:
            df: 日志DataFrame
            labels: 聚类标签
            
        Returns:
            异常日志DataFrame
        """
        logger.info("正在检测异常日志...")
        
        # 方法1: 噪声点(-1)被视为异常
        noise_logs = df[labels == -1]
        
        # 方法2: 特别小的簇可能是异常
        # 计算每个簇的大小
        cluster_sizes = Counter(labels)
        # 找出小簇(排除噪声点)
        small_clusters = [c for c, size in cluster_sizes.items() if size <= 3 and c != -1]
        small_cluster_mask = np.isin(labels, small_clusters)
        small_cluster_logs = df[small_cluster_mask]
        
        # 方法3: 主要是ERROR级别的簇
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = labels
        
        error_clusters = []
        unique_labels = sorted(set(labels))
        for label in unique_labels:
            if label == -1:
                continue
                
            cluster_logs = df_with_clusters[df_with_clusters['cluster'] == label]
            if 'level' in cluster_logs.columns:
                level_counts = cluster_logs['level'].value_counts().to_dict()
                # 如果ERROR日志占比超过50%，则视为异常簇
                error_count = level_counts.get('ERROR', 0)
                if error_count > 0 and error_count / len(cluster_logs) > 0.5:
                    error_clusters.append(label)
        
        error_cluster_mask = np.isin(labels, error_clusters)
        error_cluster_logs = df[error_cluster_mask]
        
        # 合并所有可能的异常日志
        anomaly_logs = pd.concat([noise_logs, small_cluster_logs, error_cluster_logs]).drop_duplicates()
        
        logger.info(f"检测到 {len(anomaly_logs)} 条可能的异常日志")
        return anomaly_logs
    
    def generate_analysis_report(self, df, labels, embeddings, output_dir='.'):
        """
        生成全面的分析报告
        
        Args:
            df: 日志DataFrame
            labels: 聚类标签
            embeddings: 日志嵌入向量
            output_dir: 输出目录
        """
        logger.info("正在生成分析报告...")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. 可视化聚类结果
        self.visualize_clusters(
            embeddings, 
            labels, 
            output_file=os.path.join(output_dir, 'log_clusters_2d.png')
        )
        
        # 2. 分析簇模式和特征
        cluster_analysis = self.analyze_cluster_patterns(df, labels, output_dir)
        
        # 3. 检测异常
        anomaly_logs = self.detect_anomalies(df, labels)
        anomaly_logs.to_csv(os.path.join(output_dir, 'anomaly_logs.csv'), index=False)
        
        # 4. 绘制日志级别分布
        if 'level' in df.columns:
            plt.figure(figsize=(10, 6))
            level_counts = df['level'].value_counts()
            sns.barplot(x=level_counts.index, y=level_counts.values)
            plt.title('日志级别分布')
            plt.ylabel('数量')
            plt.xlabel('日志级别')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'log_level_distribution.png'))
            plt.close()
        
        # 5. 每个簇的日志级别分布
        df_with_clusters = df.copy()
        df_with_clusters['cluster'] = labels
        
        if 'level' in df.columns:
            cluster_level_data = []
            for cluster in sorted(set(labels)):
                if cluster == -1:  # 跳过噪声
                    continue
                    
                cluster_df = df_with_clusters[df_with_clusters['cluster'] == cluster]
                level_counts = cluster_df['level'].value_counts().to_dict()
                
                for level, count in level_counts.items():
                    cluster_level_data.append({
                        'cluster': f'Cluster {cluster}',
                        'level': level,
                        'count': count
                    })
            
            if cluster_level_data:
                cluster_level_df = pd.DataFrame(cluster_level_data)
                plt.figure(figsize=(12, 8))
                sns.barplot(data=cluster_level_df, x='cluster', y='count', hue='level')
                plt.title('每个簇的日志级别分布')
                plt.ylabel('数量')
                plt.xlabel('簇')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'cluster_level_distribution.png'))
                plt.close()
        
        logger.info(f"分析报告已生成到目录: {output_dir}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='非结构化日志分析工具')
    parser.add_argument('--log_file', type=str, required=True, help='日志文件路径')
    parser.add_argument('--model_path', type=str, default='../pretrained', help='BigLog模型路径')
    parser.add_argument('--output_dir', type=str, default='./analysis_results', help='输出目录')
    parser.add_argument('--eps', type=float, default=0.5, help='DBSCAN的eps参数')
    parser.add_argument('--min_samples', type=int, default=5, help='DBSCAN的min_samples参数')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化分析器
    analyzer = LogAnalyzer(model_path=args.model_path)
    
    # 预处理日志
    df = analyzer.preprocess_logs(args.log_file)
    
    # 提取日志嵌入向量
    log_texts = df['raw'].tolist()
    embeddings = analyzer.extract_log_embeddings(log_texts)
    
    # 聚类日志
    labels = analyzer.cluster_logs(embeddings, eps=args.eps, min_samples=args.min_samples)
    
    # 生成分析报告
    analyzer.generate_analysis_report(df, labels, embeddings, args.output_dir)
    
    logger.info("日志分析完成！")

if __name__ == "__main__":
    main() 