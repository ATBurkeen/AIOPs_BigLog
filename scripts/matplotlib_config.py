#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Matplotlib中文支持配置模块
用于解决Windows环境下Matplotlib中文字体显示问题
"""

import os
import sys
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def configure_matplotlib_chinese():
    """配置Matplotlib以支持中文显示"""
    system = platform.system()
    
    if system == 'Windows':
        # Windows系统字体路径
        font_dirs = [
            'C:\\Windows\\Fonts',
            os.path.join(os.environ['WINDIR'], 'Fonts'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')  # 脚本目录下的fonts文件夹
        ]
        
        # 常用中文字体列表，按优先级排序
        chinese_fonts = [
            'Microsoft YaHei', 'SimHei', 'SimSun', 'FangSong', 'KaiTi', 'NSimSun',
            'Microsoft YaHei UI', 'SimSun-ExtB', 'MingLiU', 'PMingLiU',
            'msyh.ttf', 'simhei.ttf', 'simsun.ttc', 'fangsong.ttf', 'kaiti.ttf',
            'msyhbd.ttf', 'simfang.ttf', 'simkai.ttf'
        ]
        
    elif system == 'Linux':
        # Linux系统字体路径
        font_dirs = [
            '/usr/share/fonts',
            '/usr/local/share/fonts',
            os.path.expanduser('~/.fonts'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
        ]
        
        # Linux下常用中文字体
        chinese_fonts = [
            'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'AR PL UMing CN',
            'AR PL UKai CN', 'SimHei', 'SimSun', 'NSimSun', 'FangSong'
        ]
        
    elif system == 'Darwin':
        # macOS系统字体路径
        font_dirs = [
            '/System/Library/Fonts',
            '/Library/Fonts',
            os.path.expanduser('~/Library/Fonts'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
        ]
        
        # macOS下常用中文字体
        chinese_fonts = [
            'STHeiti', 'STSong', 'STFangsong', 'STKaiti', 
            'PingFang SC', 'Heiti SC', 'Hiragino Sans GB',
            'Hiragino Sans CNS', 'Microsoft YaHei'
        ]
    else:
        logger.warning(f"未知操作系统: {system}，无法配置中文字体")
        return False
    
    # 尝试查找系统中可用的中文字体
    font_files = []
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            font_files.extend(fm.findSystemFonts(fontpaths=[font_dir]))
    
    # 尝试使用字体列表中的字体
    font_found = False
    for font_file in font_files:
        try:
            font_name = fm.FontProperties(fname=font_file).get_name()
            if any(cf.lower() in font_name.lower() for cf in chinese_fonts) or any(cf.lower() in os.path.basename(font_file).lower() for cf in chinese_fonts):
                logger.info(f"找到中文字体: {font_name} ({font_file})")
                plt.rcParams['font.family'] = font_name
                font_found = True
                break
        except Exception as e:
            continue
    
    # 如果没有找到中文字体，尝试使用中文字体名称
    if not font_found:
        for chinese_font in chinese_fonts:
            try:
                plt.rcParams['font.family'] = chinese_font
                logger.info(f"尝试使用字体: {chinese_font}")
                font_found = True
                break
            except Exception:
                continue
    
    # 通用设置
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    if font_found:
        logger.info("Matplotlib中文字体配置成功")
        return True
    else:
        logger.warning("无法找到合适的中文字体，图表中文可能显示为方框")
        
        # 创建自定义字体目录
        fonts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
        if not os.path.exists(fonts_dir):
            try:
                os.makedirs(fonts_dir)
                logger.info(f"已创建字体目录: {fonts_dir}")
                logger.info(f"请将中文字体文件(.ttf/.ttc)复制到此目录，然后重新运行程序")
            except Exception as e:
                logger.error(f"创建字体目录失败: {str(e)}")
        
        return False

if __name__ == "__main__":
    configure_matplotlib_chinese() 