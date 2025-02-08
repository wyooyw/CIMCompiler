import os

def to_abs_path(path, parent=os.getcwd()):
    if path is None:
        return None
    if not os.path.isabs(path):
        return os.path.join(parent, path)
    return path

def show_args(args):
    s = "Arguments:\n"
    for key, value in vars(args).items():
        s += f"  {key}: {value}\n"
    return s