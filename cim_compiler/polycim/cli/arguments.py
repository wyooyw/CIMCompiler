import copy

_args = None


def set_args(args):
    global _args
    _args = copy.deepcopy(args)


def get_args():
    global _args
    return copy.deepcopy(_args)
