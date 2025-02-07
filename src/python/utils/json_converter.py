import argparse
import json

def parse_list_return_list(node):
    result = []
    for i in node:
        new_dict = parse_dict(i)
        if new_dict is not None:
            result.append(new_dict)
    return result

def parse_list(node):
    if len(node) == 1:
        return parse_dict(node[0], return_text=True)
    else:
        result = dict()
        for i in node:
            new_dict = parse_dict(i)
            if new_dict is not None:
                result.update(new_dict)
        return result

def parse_dict(node ,return_text=False):
    assert type(node) == dict
    if len(node) == 1:
        name = list(node.keys())[0]
        if name.endswith("list"):
            value = parse_list_return_list(node[name])
        else:
            value = parse_list(node[name])
        return {name: value}
    elif return_text and "text" in node:
        return node["text"]
    else:
        return None

def main(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    new_data = parse_dict(data)
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=2)

if __name__ == "__main__":
    input_path = "/home/wangyiou/project/cim_compiler_frontend/playground/.result/ast.json"
    output_path = "/home/wangyiou/project/cim_compiler_frontend/playground/ast_new.json"
    main(input_path, output_path)
