import json

def dumps_list_of_dict(data):
    assert type(data) == list
    assert all(isinstance(d, dict) for d in data)
    content = "[\n"
    for i, d in enumerate(data):
        content += "\t" + json.dumps(d)
        if i < len(data) - 1:
            content += ",\n"
        elif i == len(data) - 1:
            content += "\n"
        else:
            assert False
    content += "]"
    return content
    
def dumps_dict_of_list_of_dict(data):
    assert type(data) == dict
    assert all(isinstance(v, list) for v in data.values())
    assert all(all(isinstance(d, dict) for d in v) for v in data.values())
    content = "{\n"
    for i, (k, v) in enumerate(data.items()):
        content += f"\t\"{k}\": [\n"

        for j, d in enumerate(v):
            content += "\t\t" + json.dumps(d)
            content += "\n" if j == len(v) - 1 else ",\n"
        content += f"\t]"
        content += "\n" if i == len(data) - 1 else ",\n"
    content += "}"
    return content

if __name__ == "__main__":
    data = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    print(dumps_list_of_dict(data))
    print("-" * 100)
    data = {
        "a": [{"a": 1, "b": 2}, {"a": 3, "b": 4}],
        "b": [{"a": 5, "b": 6}, {"a": 7, "b": 8}]
    }
    print(dumps_dict_of_list_of_dict(data))
