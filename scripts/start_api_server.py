#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BigLog运维知识库API服务启动脚本
提供简单的命令行接口，用于启动API服务
"""

import os
import sys
import argparse
from ops_knowledge_api import app, initialize_knowledge_base, set_model_path

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="启动BigLog运维知识库API服务")
    parser.add_argument('--host', default='0.0.0.0', help='API服务主机地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='API服务端口 (默认: 5000)')
    parser.add_argument('--model-path', default='../pretrained', help='BigLog预训练模型路径 (默认: ../pretrained)')
    parser.add_argument('--debug', action='store_true', help='启用debug模式')
    return parser.parse_args()

def check_model_path(model_path):
    """检查模型路径是否有效"""
    if not os.path.exists(model_path):
        print(f"错误: 模型路径 '{model_path}' 不存在")
        return False
    
    required_files = ['pytorch_model.bin', 'config.json', 'vocab.txt']
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            print(f"错误: 在模型路径中未找到必要的文件 '{file}'")
            return False
    
    return True

def main():
    """主函数"""
    args = parse_arguments()
    
    # 确保模型路径存在
    if not check_model_path(args.model_path):
        sys.exit(1)
    
    # 设置模型路径
    model_path = os.path.abspath(args.model_path)
    set_model_path(model_path)
    
    # 初始化知识库
    try:
        initialize_knowledge_base()
    except Exception as e:
        print(f"初始化知识库时出错: {str(e)}")
        sys.exit(1)
    
    # 显示启动信息
    print(f"\n{'=' * 60}")
    print(f"BigLog运维知识库API服务正在启动")
    print(f"{'=' * 60}")
    print(f"服务地址: http://{args.host}:{args.port}")
    print(f"API文档: http://{args.host}:{args.port}/")
    print(f"使用模型: {model_path}")
    print(f"{'=' * 60}\n")
    
    try:
        # 启动Flask应用
        app.run(host=args.host, port=args.port, debug=args.debug)
    except KeyboardInterrupt:
        print("\n服务已通过Ctrl+C终止")
    except Exception as e:
        print(f"\n服务启动失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 