#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
启动BigLog统一测试平台
"""

import os
import sys
import subprocess
import webbrowser
import time

def main():
    print("=" * 60)
    print("BigLog 统一测试平台")
    print("=" * 60)
    
    # 检查当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    print(f"当前工作目录: {current_dir}")
    
    # 检查必要的文件
    required_files = [
        'unified_web_app.py',
        'templates/index.html',
        'static/style.css'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ 缺少必要文件:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n请确保所有文件都已创建。")
        return
    
    print("✅ 所有必要文件检查通过")
    
    # 检查依赖
    try:
        import flask
        import torch
        import transformers
        import sklearn
        import matplotlib
        import numpy
        import pandas
        print("✅ 所有依赖包检查通过")
    except ImportError as e:
        print(f"❌ 缺少依赖包: {e}")
        print("请运行: pip install flask torch transformers scikit-learn matplotlib numpy pandas")
        return
    
    # 检查预训练模型
    model_path = "../pretrained"
    if not os.path.exists(model_path):
        print(f"⚠️  预训练模型目录不存在: {model_path}")
        print("应用将以有限功能模式运行")
    else:
        print("✅ 预训练模型检查通过")
    
    # 创建必要的目录
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('analysis_results', exist_ok=True)
    print("✅ 目录结构检查通过")
    
    # 启动Web应用
    print("\n🚀 启动BigLog统一测试平台...")
    print("访问地址: http://localhost:5000")
    print("按 Ctrl+C 停止服务")
    print("-" * 60)
    
    try:
        # 延迟打开浏览器
        def open_browser():
            time.sleep(2)
            webbrowser.open('http://localhost:5000')
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # 启动Flask应用
        from unified_web_app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("请检查错误信息并重试")

if __name__ == '__main__':
    main() 