import logging
import os
def get_log_level_from_env():
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    return {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }[log_level]

def get_logger(name, output_file=None):
    # 创建日志记录器，设置日志名称
    logger = logging.getLogger(name)
    level = get_log_level_from_env()
    logger.setLevel(level)  # 设置日志级别为DEBUG

    # 添加控制台处理器
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if output_file is not None:
        # 创建文件处理器，并指定日志文件路径
        file_handler = logging.FileHandler(output_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
