import os
import shutil

def keep_flat(path):
    """
    flat_code.json -> final_code.json
    flat_regs.json -> regs.json
    flat_stats.json -> stats.json
    keep global_image
    remove all other files
    """
    
    for root, dirs, files in os.walk(path):
        for file_name in files:
            if file_name not in ["flat_code.json", "flat_stats.json", "global_image", "flat_macro_ultilization.json", "flat_pimset.json"]:
                file_path = os.path.join(root, file_name)
                print(f"remove {file_path}")
                os.remove(file_path)
        rename_list = [
            ("flat_code.json", "final_code.json"),
            ("flat_stats.json", "stats.json"),
            ("flat_macro_ultilization.json", "macro_ultilization.json"),
            ("flat_pimset.json", "pimset.json"),
        ]
        for old_name, new_name in rename_list:
            old_file_path = os.path.join(root, old_name)
            if not os.path.exists(old_file_path):
                continue
            new_file_path = os.path.join(root, new_name)
            assert not os.path.exists(new_file_path), f"{new_file_path} already exists"
            os.rename(old_file_path, new_file_path)
            print(f"rename {old_file_path} to {new_file_path}")

def keep_origin(path):
    """
    keep global_image, final_code.json, regs.json, stats.json
    remove all other files
    """
    
    for root, dirs, files in os.walk(path):
        for file_name in files:
            if file_name not in ["final_code.json", "stats.json", "global_image"]:
                file_path = os.path.join(root, file_name)
                print(f"remove {file_path}")
                os.remove(file_path)

def main():
    path = "/home/wangyiou/project/cim_compiler_frontend/playground/.result/2024-09-10T17-49-56-small-input-0.4"
    keep_flat(path)
    # keep_origin(path)

if __name__=="__main__":
    main()