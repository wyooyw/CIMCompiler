from data_type import *

class MacroConfig:
    def __init__(self, n_macro, n_row, n_comp, n_bcol):
        self.n_macro = n_macro
        self.n_row = n_row
        self.n_comp = n_comp
        self.n_bcol = n_bcol

    def n_vcol(self, bitwidth):
        assert self.n_bcol % bitwidth == 0
        return self.n_bcol // bitwidth

class MacroUtil:
    def __init__(self, macro_memory, macro_config):
        self.macro_memory = macro_memory
        self.macro_config = macro_config
        
    def extract_macro_structure_from_memory(self, data_type):
        """
        <N_ROW, N_COMP, N_MACRO, N_VCOL>
        """
        if type(data_type)==int:
            bitwidth = data_type
            data_type = get_dtype_from_bitwidth(bitwidth)
        else:
            bitwidth = get_bitwidth_from_dtype(data_type)
        data_bytes = self.macro_memory.read_all()
        macro = np.frombuffer(data_bytes, dtype=data_type)
        macro = macro.reshape(self.macro_config.n_row, 
                            self.macro_config.n_comp, 
                            self.macro_config.n_macro, 
                            self.macro_config.n_vcol(bitwidth))
        return macro

    def get_macro_data(self, 
        activate_row,
        data_type,
        activate_element_row_num,
        activate_element_col_num):
        
        macro = self.extract_macro_structure_from_memory(data_type).reshape(*macro.shape[:2], -1)
        data = macro[activate_row, :activate_element_row_num, :activate_element_col_num]
        return data