from polycim.passes.base import BreadthFirstPass


class FilterSingleOpPass(BreadthFirstPass):
    def __init__(self, n_keep=1):
        super().__init__()
        self.op_list = list()
        self.n_keep = n_keep

    def apply(self, operator):
        self.op_list.append(operator)

    def apply_all(self):
        pass

    def get_result(self):
        n_keep = min(self.n_keep, len(self.op_list))
        return self.op_list[:n_keep]
