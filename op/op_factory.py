class OpFactory:
    def __init__(self, cim_config):
        self.cim_config = cim_config
        self.op_list = []

    def register_op(self, op_config):
        self.op_list.append(op_config,self.cim_config)

    def select_op(self, op_config):
        for op in self.op_list:
            if op.is_suitable(op_config, self.cim_config):
                return op
        assert False, f"No suitable operator for {op_config=}"
        return None

OP_FACTORY = OpFactory()