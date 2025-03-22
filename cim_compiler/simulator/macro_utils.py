import json

from cim_compiler.simulator.data_type import *
from cim_compiler.utils.logger import get_logger

logger = get_logger(__name__)

class MacroConfig:
    def __init__(self, n_macro, n_row, n_comp, n_bcol, n_group):
        self.n_macro = n_macro
        self.n_row = n_row
        self.n_comp = n_comp
        self.n_bcol = n_bcol
        self.n_group = n_group
        self.n_macro_per_group = self.n_macro // self.n_group
        logger.debug(f"Macro config: {n_macro=}, {n_row=}, {n_comp=}, {n_bcol=}, {n_group=}")

    def n_vcol(self, bitwidth):
        assert self.n_bcol % bitwidth == 0
        return self.n_bcol // bitwidth

    def n_group_vcol(self, bitwidth):
        return self.n_vcol(bitwidth) * self.n_macro_per_group

    def total_size(self):
        return self.n_macro * self.n_row * self.n_comp * self.n_bcol // 8

    @classmethod
    def from_config(
        cls,
        config_path,
    ):
        with open(config_path, "r") as f:
            config = json.load(f)
        n_macro = config["macro"]["n_macro"]
        n_row = config["macro"]["n_row"]
        n_comp = config["macro"]["n_comp"]
        n_bcol = config["macro"]["n_bcol"]
        n_group = config["macro"]["n_group"]
        return cls(n_macro, n_row, n_comp, n_bcol, n_group)


class MacroUtil:
    def __init__(self, macro_memory, macro_config):
        self.macro_memory = macro_memory
        self.macro_config = macro_config

    def extract_macro_structure_from_memory(self, data_type):
        """
        <N_ROW, N_COMP, N_GROUP, N_MACRO_PER_GROUP, N_VCOL>
        """
        if type(data_type) == int:
            bitwidth = data_type
            data_type = get_dtype_from_bitwidth(bitwidth)
        elif type(data_type) == np.int32:
            bitwidth = data_type.item()
            data_type = get_dtype_from_bitwidth(bitwidth)
        elif data_type in [np.int8, np.int16, np.int32, np.float16, np.float32]:
            bitwidth = get_bitwidth_from_dtype(data_type)
        else:
            assert False, f"Unsupport {data_type=}"

        data_bytes = self.macro_memory.read_all()
        macro = np.frombuffer(data_bytes, dtype=data_type)
        macro = macro.reshape(
            self.macro_config.n_row,
            self.macro_config.n_comp,
            self.macro_config.n_group,
            self.macro_config.n_macro_per_group,
            self.macro_config.n_vcol(bitwidth),
        )
        return macro

    def get_macro_data(
        self,
        activate_row,
        data_type,
        activate_element_row_num,
        activate_element_col_num,
        activate_group_num,
    ):

        assert (
            0 <= activate_group_num and activate_group_num <= self.macro_config.n_group
        ), f"{activate_group_num=}, {self.macro_config.n_group=}"

        macro = self.extract_macro_structure_from_memory(data_type)
        # print(f"{macro.shape=}")
        macro = macro.reshape(*macro.shape[:3], -1)
        data = macro[activate_row, :, :activate_group_num, :activate_element_col_num]
        return data
