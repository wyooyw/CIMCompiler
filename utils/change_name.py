import os
import re

def rename_directories(root_path):
    # 编译正则表达式模式：匹配数字+`+_add
    pattern = re.compile(r'(\d+)`_add$')
    
    dir_name_map = {}

    # 遍历目录
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        for dirname in dirnames:
            # 检查目录名是否匹配模式
            match = pattern.match(dirname)
            if match:
                # 构建旧路径和新路径
                old_path = os.path.join(dirpath, dirname)
                new_name = f"{match.group(1)}_add"  # 移除 ` 符号
                new_path = os.path.join(dirpath, new_name)
                dir_name_map[old_path] = new_path

                # print(f"Renamed: {old_path} -> {new_path}")
                
                # try:
                #     # 重命名目录
                #     os.rename(old_path, new_path)
                #     print(f"Renamed: {old_path} -> {new_path}")
                # except OSError as e:
                #     print(f"Error renaming {old_path}: {e}")

    for idx, (old_path, new_path) in enumerate(dir_name_map.items()):
        os.rename(old_path, new_path)
        print(f"Renamed: {idx} {old_path} -> {new_path}")

def rename_directories_2(root_path):
    """
    bit_value_sparse_0.6 -> bit_value_sparse
    bit_value_sparse_0.4 -> bit_value_sparse_0p4
    bit_value_sparse_0.2 -> bit_value_sparse_0p2
    """
    pattern = re.compile(r'bit_value_sparse_(\d+)\.(\d+)$')
    dir_name_map = {}
    for dirpath, dirnames, filenames in os.walk(root_path, topdown=False):
        for dirname in dirnames:
            match = pattern.match(dirname)
            if match:
                old_path = os.path.join(dirpath, dirname)
                if match.group(2)=="6":
                    new_name = f"bit_value_sparse"
                else:
                    new_name = f"bit_value_sparse_{match.group(1)}p{match.group(2)}"
                new_path = os.path.join(dirpath, new_name)
                dir_name_map[old_path] = new_path

    for idx, (old_path, new_path) in enumerate(dir_name_map.items()):
        os.rename(old_path, new_path)
        print(f"Renamed: {idx} {old_path} -> {new_path}")

# 使用示例
root_directory = "/home/wangyiou/project/cim_compiler_frontend/playground/.result/2024-12-19T18-55-12"  # 当前目录，你可以修改为其他路径
rename_directories_2(root_directory)