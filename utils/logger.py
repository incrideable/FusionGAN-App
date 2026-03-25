import logging
import os
import sys
from datetime import datetime
import colorlog

def setup_logger(name, log_dir=None, level=logging.INFO):
    """设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_dir: 日志文件保存目录
        level: 日志级别
        
    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 避免重复添加处理器
    if logger.handlers:
        return logger
    
    # 创建格式器
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器（带颜色）
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # 彩色格式
    color_formatter = colorlog.ColoredFormatter(
        fmt='%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(color_formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name, log_dir=None, level=logging.INFO):
    """获取日志记录器（快捷函数）"""
    return setup_logger(name, log_dir, level)

# 测试函数
if __name__ == "__main__":
    # 测试日志功能
    logger = setup_logger('TestLogger', log_dir='logs')
    
    logger.debug("这是一条调试信息")
    logger.info("这是一条普通信息")
    logger.warning("这是一条警告信息")
    logger.error("这是一条错误信息")
    logger.critical("这是一条严重错误信息")
    
    print("日志测试完成！")