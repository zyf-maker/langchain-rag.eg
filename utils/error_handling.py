# 错误处理和日志管理模块
import logging
import traceback
import sys
from datetime import datetime

# 创建日志目录
import os
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# 配置日志格式
log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")

# 设置根日志记录器
logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 创建应用特定的日志记录器
logger = logging.getLogger("rag_app")


class RAGError(Exception):
    """
    RAG应用自定义异常类
    """
    def __init__(self, message, error_type="general", original_error=None):
        self.message = message
        self.error_type = error_type
        self.original_error = original_error
        super().__init__(self.message)


def handle_error(error, show_traceback=False, custom_message=None):
    """
    统一错误处理函数
    
    Args:
        error: 异常对象
        show_traceback: 是否显示完整的堆栈跟踪
        custom_message: 自定义错误消息
    
    Returns:
        str: 格式化的错误消息
    """
    # 记录错误日志
    logger.error(f"错误类型: {type(error).__name__}")
    logger.error(f"错误消息: {str(error)}")
    logger.error(f"堆栈跟踪: {traceback.format_exc()}")
    
    # 构建错误消息
    error_message = custom_message if custom_message else str(error)
    
    # 如果是自定义的RAGError
    if isinstance(error, RAGError):
        error_message = f"[{error.error_type}] {error_message}"
    
    # 如果需要显示堆栈跟踪
    if show_traceback:
        error_message += f"\n\n堆栈跟踪:\n{traceback.format_exc()}"
    
    return error_message


def log_info(message):
    """
    记录信息日志
    
    Args:
        message: 日志消息
    """
    logger.info(message)


def log_warning(message):
    """
    记录警告日志
    
    Args:
        message: 警告消息
    """
    logger.warning(message)


def log_error(message, exc_info=False):
    """
    记录错误日志
    
    Args:
        message: 错误消息
        exc_info: 是否包含异常信息
    """
    logger.error(message, exc_info=exc_info)


def log_debug(message):
    """
    记录调试日志
    
    Args:
        message: 调试消息
    """
    logger.debug(message)


def safe_execute(func, *args, **kwargs):
    """
    安全执行函数，捕获所有异常
    
    Args:
        func: 要执行的函数
        *args: 函数参数
        **kwargs: 函数关键字参数
    
    Returns:
        tuple: (success, result/error)
    """
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        error_msg = handle_error(e)
        return False, error_msg