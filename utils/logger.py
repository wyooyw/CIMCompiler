import logging

def get_logger(name, level, output_file=None):
    # 创建日志记录器，设置日志名称
    logger = logging.getLogger(name)
    logger.setLevel(level)  # 设置日志级别为DEBUG
    if output_file is not None:
        # 创建文件处理器，并指定日志文件路径
        file_handler = logging.FileHandler(output_file)

        # 创建日志格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # 将文件处理器添加到日志记录器
        logger.addHandler(file_handler)

    return logger