#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
中文字体安装工具
自动下载和安装中文字体到fonts目录，用于解决Matplotlib中文显示问题
"""

import os
import sys
import platform
import shutil
import logging
import urllib.request
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 字体下载链接
# 注意：这些链接可能会失效，如果失效请手动下载字体
FONT_URLS = {
    'SimHei': 'https://github.com/StellarCN/scp_zh/raw/master/fonts/SimHei.ttf',
    'SimSun': 'https://github.com/micmro/Stylify-Me/raw/main/app/lib/conserve/fonts/simsun.ttc',
    'Microsoft YaHei': 'https://github.com/Zenozhouzhao/Font/raw/master/Chinese/%E5%BE%AE%E8%BD%AF%E9%9B%85%E9%BB%91.ttf'
}

def create_fonts_dir():
    """创建fonts目录"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fonts_dir = os.path.join(script_dir, 'fonts')
    
    if not os.path.exists(fonts_dir):
        try:
            os.makedirs(fonts_dir)
            logger.info(f"已创建字体目录: {fonts_dir}")
        except Exception as e:
            logger.error(f"创建字体目录失败: {str(e)}")
            return None
    
    return fonts_dir

def download_font(font_name, url, fonts_dir):
    """下载字体文件"""
    if font_name.lower() == 'simhei':
        filename = 'SimHei.ttf'
    elif font_name.lower() == 'simsun':
        filename = 'SimSun.ttc'
    elif font_name.lower() == 'microsoft yahei':
        filename = 'msyh.ttf'
    else:
        filename = f"{font_name.replace(' ', '_')}.ttf"
    
    target_path = os.path.join(fonts_dir, filename)
    
    # 如果字体文件已存在，则跳过
    if os.path.exists(target_path):
        logger.info(f"字体文件已存在: {target_path}")
        return True
    
    try:
        logger.info(f"正在下载字体: {font_name} 从 {url}")
        urllib.request.urlretrieve(url, target_path)
        logger.info(f"字体下载成功: {target_path}")
        return True
    except Exception as e:
        logger.error(f"下载字体失败: {str(e)}")
        return False

def copy_system_fonts(fonts_dir):
    """从系统字体目录复制中文字体"""
    system = platform.system()
    success = False
    
    if system == 'Windows':
        windows_fonts_dir = os.path.join(os.environ['WINDIR'], 'Fonts')
        fonts_to_copy = ['simhei.ttf', 'simsun.ttc', 'msyh.ttf', 'simfang.ttf', 'simkai.ttf']
        
        for font in fonts_to_copy:
            src_path = os.path.join(windows_fonts_dir, font)
            if os.path.exists(src_path):
                try:
                    dest_path = os.path.join(fonts_dir, font)
                    shutil.copy2(src_path, dest_path)
                    logger.info(f"已复制系统字体: {font}")
                    success = True
                except Exception as e:
                    logger.warning(f"复制字体失败 {font}: {str(e)}")
    
    elif system == 'Linux':
        linux_font_dirs = [
            '/usr/share/fonts',
            '/usr/local/share/fonts',
            os.path.expanduser('~/.fonts')
        ]
        fonts_to_copy = ['wqy-microhei.ttc', 'wqy-zenhei.ttc']
        
        for font_dir in linux_font_dirs:
            if os.path.exists(font_dir):
                for root, dirs, files in os.walk(font_dir):
                    for font in files:
                        if font.lower() in [f.lower() for f in fonts_to_copy]:
                            try:
                                src_path = os.path.join(root, font)
                                dest_path = os.path.join(fonts_dir, font)
                                shutil.copy2(src_path, dest_path)
                                logger.info(f"已复制系统字体: {font}")
                                success = True
                            except Exception as e:
                                logger.warning(f"复制字体失败 {font}: {str(e)}")
    
    elif system == 'Darwin':  # macOS
        mac_font_dirs = [
            '/System/Library/Fonts',
            '/Library/Fonts',
            os.path.expanduser('~/Library/Fonts')
        ]
        fonts_to_copy = ['STHeiti Light.ttc', 'STHeiti Medium.ttc', 'PingFang.ttc']
        
        for font_dir in mac_font_dirs:
            if os.path.exists(font_dir):
                for font in fonts_to_copy:
                    src_path = os.path.join(font_dir, font)
                    if os.path.exists(src_path):
                        try:
                            dest_path = os.path.join(fonts_dir, font)
                            shutil.copy2(src_path, dest_path)
                            logger.info(f"已复制系统字体: {font}")
                            success = True
                        except Exception as e:
                            logger.warning(f"复制字体失败 {font}: {str(e)}")
    
    return success

def main():
    """主函数"""
    # 创建fonts目录
    fonts_dir = create_fonts_dir()
    if not fonts_dir:
        return
    
    # 首先尝试从系统复制字体
    logger.info("尝试从系统字体目录复制中文字体...")
    copy_success = copy_system_fonts(fonts_dir)
    
    # 如果系统复制失败，尝试从网络下载字体
    if not copy_success:
        logger.info("尝试从网络下载中文字体...")
        for font_name, url in FONT_URLS.items():
            download_font(font_name, url, fonts_dir)
    
    # 显示安装结果
    installed_fonts = [f for f in os.listdir(fonts_dir) if f.endswith(('.ttf', '.ttc', '.otf'))]
    
    if installed_fonts:
        logger.info(f"成功安装了 {len(installed_fonts)} 个字体:")
        for font in installed_fonts:
            logger.info(f"  - {font}")
        logger.info("请重新运行您的脚本，中文显示问题应该已解决。")
    else:
        logger.warning("未能安装任何字体。请手动将中文字体文件复制到以下目录:")
        logger.warning(f"  {fonts_dir}")
        logger.warning("推荐的字体有: SimHei.ttf, SimSun.ttc, msyh.ttf (微软雅黑)")

if __name__ == "__main__":
    main() 