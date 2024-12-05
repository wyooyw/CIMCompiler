import json
import os
import argparse

def make_config(args):
    """
    1.read config_template.json
    2.calculate each memory's offset, fill into 'offset_byte'
    3.write config.json
    """
    import json

    config_template_path = args.config_template_path
    # if it is relative path, convert it to absolute path
    if not os.path.isabs(config_template_path):
        config_template_path = os.path.join(os.path.dirname(__file__), config_template_path)

    config_path = os.path.join(os.path.dirname(__file__), "config.json")
    with open(config_template_path, "r") as f:
        config = json.load(f)
    last_end = 0
    for memory in config["memory_list"]:
        addressing = memory["addressing"]
        addressing["offset_byte"] = last_end
        last_end += addressing["size_byte"]
    
    print(f"config template path: {config_template_path}")
    if os.path.exists(config_path):
        make_sure = input(
            f"config.json already exists, do you want to overwrite it? (y/n)"
        )
    if make_sure.lower() == "y":
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)
        print("make config.json done")

def parse_args():
    parser = argparse.ArgumentParser(description="make config.json")
    parser.add_argument("--config-template-path", "-p", type=str, help="config template path")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    make_config(args)
