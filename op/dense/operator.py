class OperatorMaker:
    def __init__(self, cim_config):
        self.cim_config = cim_config

    def is_suitable(self, op_config, cim_config):
        assert False, "Not implemented"

    def get_code(self, op_config, cim_config):
        assert False, "Not implemented"

    def get_image(self, op_config, cim_config):
        assert False, "Not implemented"

    def get_code_and_image(self):
        return self.get_code(), self.get_image()

class Conv2dMaker(OperatorMaker):
    def __init__(self, cim_config):
        super().__init__(cim_config)

    def is_suitable(self, op_config, ):
        """
        n_row * n_col * n_comp * n_out * n_in <= 1024
        """
        return cim_config.n_row * cim_config.n_col * cim_config.n_comp * cim_config.n_out * cim_config.n_in <= 1024

class Conv2dShortReductionOperatorMaker(Conv2dMaker):
    def __init__(self):
        pass

    def is_suitable(self, op_config, cim_config):
        """
        reduce axis is shorter than n_comp * n_row
        """
        return super().is_suitable(op_config, cim_config) and op_config.reduce_axis_len <= cim_config.n_comp * cim_config.n_row

class Conv2dLongReductionOperatorMaker(Conv2dMaker):
    def __init__(self):
        pass

    def is_suitable(self, op_config, cim_config):
        """
        reduce axis is longer than n_comp * n_row
        """
        return super().is_suitable(op_config, cim_config) and op_config.reduce_axis_len > cim_config.n_comp * cim_config.n_row

OP_FACTORY.register_op(Conv2dShortReductionOperator)
OP_FACTORY.register_op(Conv2dLongReductionOperator)