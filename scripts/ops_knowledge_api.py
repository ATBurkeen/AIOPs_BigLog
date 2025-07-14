#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
import numpy as np
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

class OpsKnowledgeBase:
    """运维知识库类"""
    
    def __init__(self, model_path='../pretrained', knowledge_file='knowledge_base.json'):
        """
        初始化运维知识库
        
        Args:
            model_path: BigLog预训练模型路径
            knowledge_file: 知识库文件路径
        """
        self.model_path = model_path
        self.knowledge_file = knowledge_file
        
        # 加载模型和分词器
        logger.info(f"正在加载BigLog模型: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = BertModel.from_pretrained(model_path)
        self.model.eval()
        
        # 加载知识库
        self.knowledge_base = self._load_knowledge_base()
        
        # 预计算知识库条目的嵌入向量
        logger.info("正在预计算知识库嵌入向量...")
        self.knowledge_embeddings = self._precompute_embeddings()
        
    def _load_knowledge_base(self):
        """加载知识库文件"""
        if os.path.exists(self.knowledge_file):
            logger.info(f"正在加载知识库文件: {self.knowledge_file}")
            try:
                with open(self.knowledge_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"知识库文件 {self.knowledge_file} 格式错误，将创建新的知识库")
                return {
                    "version": "1.0",
                    "entries": []
                }
        else:
            logger.warning(f"知识库文件不存在: {self.knowledge_file}，将创建新的知识库")
            return {
                "version": "1.0",
                "entries": []
            }
    
    def save_knowledge_base(self):
        """保存知识库到文件"""
        try:
            with open(self.knowledge_file, 'w', encoding='utf-8') as f:
                json.dump(self.knowledge_base, f, ensure_ascii=False, indent=2)
            logger.info(f"知识库已保存到: {self.knowledge_file}")
        except Exception as e:
            logger.error(f"保存知识库文件时出错: {str(e)}")
    
    def _precompute_embeddings(self):
        """预计算知识库条目的嵌入向量"""
        entries = self.knowledge_base.get("entries", [])
        if not entries:
            return np.array([])
        
        # 提取问题文本
        questions = [entry["question"] for entry in entries]
        
        # 计算嵌入向量
        return self.get_embeddings(questions)
    
    def get_embeddings(self, texts, max_length=150):
        """
        计算文本的嵌入向量
        
        Args:
            texts: 文本列表
            max_length: 最大序列长度
            
        Returns:
            文本嵌入向量
        """
        # 对文本进行分词
        inputs = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # 计算嵌入向量
        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用[CLS]令牌的表示作为整个文本的表示
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings
    
    def add_entry(self, question, answer, category=None, tags=None):
        """
        添加知识库条目
        
        Args:
            question: 问题
            answer: 答案
            category: 类别
            tags: 标签列表
        
        Returns:
            新条目的ID
        """
        entries = self.knowledge_base.get("entries", [])
        
        # 创建新条目
        entry_id = len(entries)
        new_entry = {
            "id": entry_id,
            "question": question,
            "answer": answer,
            "category": category or "general",
            "tags": tags or []
        }
        
        # 添加到知识库
        entries.append(new_entry)
        self.knowledge_base["entries"] = entries
        
        # 更新嵌入向量
        if len(self.knowledge_embeddings) == 0:
            self.knowledge_embeddings = self.get_embeddings([question])
        else:
            question_embedding = self.get_embeddings([question])
            self.knowledge_embeddings = np.vstack([self.knowledge_embeddings, question_embedding])
        
        # 保存知识库
        self.save_knowledge_base()
        
        return entry_id
    
    def search(self, query, top_k=3, threshold=0.5):
        """
        搜索知识库
        
        Args:
            query: 查询文本
            top_k: 返回的最大结果数量
            threshold: 相似度阈值
            
        Returns:
            匹配的知识库条目列表
        """
        entries = self.knowledge_base.get("entries", [])
        if not entries:
            return []
        
        # 计算查询文本的嵌入向量
        query_embedding = self.get_embeddings([query])
        
        # 计算与知识库条目的相似度
        similarities = cosine_similarity(query_embedding, self.knowledge_embeddings)[0]
        
        # 获取相似度高于阈值的条目
        results = []
        for i, sim in enumerate(similarities):
            if sim >= threshold:
                entry = entries[i].copy()
                entry["similarity"] = float(sim)
                results.append(entry)
        
        # 按相似度排序并返回top_k个结果
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

# 全局知识库实例和模型路径
knowledge_base = None
MODEL_PATH = '../pretrained'

def set_model_path(model_path):
    """设置模型路径"""
    global MODEL_PATH
    MODEL_PATH = model_path
    logger.info(f"已设置模型路径为: {MODEL_PATH}")

def initialize_knowledge_base():
    """初始化知识库实例"""
    global knowledge_base
    if knowledge_base is None:
        logger.info("初始化知识库实例...")
        knowledge_base = OpsKnowledgeBase(model_path=MODEL_PATH)
        
        # 添加示例条目
        if not knowledge_base.knowledge_base.get("entries"):
            logger.info("添加示例知识条目...")
            sample_entries = [
                {
                    "question": "MySQL数据库连接失败怎么处理？",
                    "answer": "1. 检查数据库服务是否运行\n2. 验证IP地址和端口是否正确\n3. 确认用户名和密码是否正确\n4. 检查防火墙设置\n5. 查看数据库日志获取详细错误信息",
                    "category": "database",
                    "tags": ["MySQL", "连接", "故障排除"]
                },
                {
                    "question": "如何解决服务器CPU使用率过高的问题？",
                    "answer": "1. 使用top命令查看占用CPU的进程\n2. 使用ps命令获取详细进程信息\n3. 检查是否存在异常进程\n4. 考虑优化应用程序代码\n5. 评估是否需要升级硬件资源",
                    "category": "performance",
                    "tags": ["CPU", "性能", "监控"]
                },
                {
                    "question": "如何进行日志分析以发现系统异常？",
                    "answer": "1. 使用grep、awk等工具筛选关键日志\n2. 查找ERROR、WARN等级别的日志条目\n3. 分析日志中的时间戳寻找故障点\n4. 使用日志分析工具如ELK进行可视化分析\n5. 建立日志告警机制及时发现问题",
                    "category": "logging",
                    "tags": ["日志", "分析", "故障排除"]
                }
            ]
            
            for entry in sample_entries:
                knowledge_base.add_entry(
                    entry["question"],
                    entry["answer"],
                    entry["category"],
                    entry["tags"]
                )
        
        logger.info(f"知识库初始化完成，包含 {len(knowledge_base.knowledge_base.get('entries', []))} 条知识条目")
    else:
        logger.info("知识库实例已初始化")

@app.route('/', methods=['GET'])
def api_root():
    """API根路径，返回API文档"""
    return """
    <html>
        <head>
            <title>BigLog运维知识库API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                h1 { color: #333; }
                h2 { color: #444; margin-top: 30px; }
                pre { background: #f4f4f4; padding: 15px; border-radius: 5px; }
                .endpoint { background: #e9f7fe; padding: 10px; border-left: 4px solid #2196F3; margin-bottom: 20px; }
                .method { font-weight: bold; color: #2196F3; }
            </style>
        </head>
        <body>
            <h1>BigLog运维知识库API</h1>
            <p>这是基于BigLog预训练模型的运维知识库API服务，用于智能运维中的知识检索和管理。</p>
            
            <h2>API端点说明</h2>
            
            <div class="endpoint">
                <p><span class="method">POST</span> /api/search</p>
                <p>搜索知识库，根据查询文本返回相关的知识条目。</p>
                <p>请求示例:</p>
                <pre>
{
  "query": "数据库连接失败",
  "top_k": 3,
  "threshold": 0.5
}
                </pre>
                <p>响应示例:</p>
                <pre>
{
  "query": "数据库连接失败",
  "results": [
    {
      "id": 0,
      "question": "MySQL数据库连接失败怎么处理？",
      "answer": "1. 检查数据库服务是否运行\\n2. 验证IP地址和端口是否正确\\n3. 确认用户名和密码是否正确\\n4. 检查防火墙设置\\n5. 查看数据库日志获取详细错误信息",
      "category": "database",
      "tags": ["MySQL", "连接", "故障排除"],
      "similarity": 0.92
    }
  ],
  "count": 1
}
                </pre>
            </div>
            
            <div class="endpoint">
                <p><span class="method">POST</span> /api/add</p>
                <p>添加知识库条目。</p>
                <p>请求示例:</p>
                <pre>
{
  "question": "如何排查网络连接问题?",
  "answer": "1. 检查网络配置\\n2. 使用ping命令测试连通性\\n3. 检查防火墙设置\\n4. 查看路由表\\n5. 检查DNS设置",
  "category": "network",
  "tags": ["网络", "连接", "故障排除"]
}
                </pre>
                <p>响应示例:</p>
                <pre>
{
  "success": true,
  "id": 3,
  "message": "知识库条目添加成功"
}
                </pre>
            </div>
            
            <h2>使用工具</h2>
            <p>您可以使用以下工具测试API:</p>
            <ul>
                <li>Postman</li>
                <li>curl</li>
                <li>浏览器开发者工具</li>
            </ul>
            <p>curl示例:</p>
            <pre>curl -X POST http://localhost:5000/api/search -H "Content-Type: application/json" -d '{"query": "数据库连接失败"}'</pre>
        </body>
    </html>
    """

@app.route('/api/search', methods=['POST'])
def search_knowledge():
    """搜索知识库API"""
    try:
        data = request.json
        query = data.get('query')
        top_k = data.get('top_k', 3)
        threshold = data.get('threshold', 0.5)
        
        if not query:
            return jsonify({"error": "查询内容不能为空"}), 400
        
        results = knowledge_base.search(query, top_k=top_k, threshold=threshold)
        return jsonify({
            "query": query,
            "results": results,
            "count": len(results)
        })
    except Exception as e:
        logger.error(f"搜索知识库时出错: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/api/add', methods=['POST'])
def add_knowledge():
    """添加知识库条目API"""
    try:
        data = request.json
        question = data.get('question')
        answer = data.get('answer')
        category = data.get('category')
        tags = data.get('tags', [])
        
        if not question or not answer:
            return jsonify({"error": "问题和答案不能为空"}), 400
        
        entry_id = knowledge_base.add_entry(question, answer, category, tags)
        return jsonify({
            "success": True,
            "id": entry_id,
            "message": "知识库条目添加成功"
        })
    except Exception as e:
        logger.error(f"添加知识库条目时出错: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def main():
    """主函数"""
    # 初始化知识库
    initialize_knowledge_base()
    
    # 启动API服务
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    main() 