from utils.df_layout import tensor_int8_to_bits
from utils.bit_sparse_weight_transform import recover_csd, csd_to_int
import numpy as np
class MetaUtil:
    def __init__(self, meta_memory, macro_config):
        self.meta_memory = meta_memory
        self.macro_config = macro_config

    def get_meta(self, meta_addr, n_group):
        n_macro_per_group = self.macro_config.n_macro // n_group
        info_size = self.macro_config.n_comp * n_macro_per_group * self.macro_config.n_bcol * 3 // 8
        info_data = self.meta_memory.read(meta_addr, info_size)
        info_tensor = np.frombuffer(info_data, dtype="int8").reshape(-1)

        info_tensor = tensor_int8_to_bits(info_tensor)
        info_tensor = info_tensor.reshape(self.macro_config.n_comp, 
                                        n_macro_per_group , 
                                        self.macro_config.n_bcol , 
                                        3)
        # info_tensor = info_tensor[:,0:macro_num, :,:]
        info_tensor = info_tensor.reshape(self.macro_config.n_comp, -1, 3)
        return info_tensor

    def recover_weight(self, meta_addr, wtensor, n_group):
        n_comp = self.macro_config.n_comp
        n_macro_per_group = self.macro_config.n_macro // n_group
        n_bcol_per_group = self.macro_config.n_bcol * n_macro_per_group

        wtensor = tensor_int8_to_bits(wtensor)
        # print(wtensor.shape)
        wtensor = wtensor.reshape(self.macro_config.n_comp, n_bcol_per_group)

        info_tensor = self.get_meta(meta_addr,n_group)
        # print(f"{info_tensor.shape=}")
        # import pdb; pdb.set_trace()

        recovered_wtensor = np.zeros((self.macro_config.n_comp, n_bcol_per_group), dtype=np.int32)
        for i_comp in range(self.macro_config.n_comp):
            for i_col_and_macro in range(n_bcol_per_group):
                val = wtensor[i_comp, i_col_and_macro]
                sign = info_tensor[i_comp, i_col_and_macro, 0]
                location = info_tensor[i_comp, i_col_and_macro, 1:3]
                # print(type(val),type(sign),type(location))
                csd = recover_csd(val, sign, location)
                int_val = csd_to_int(csd)
                # print(f"value:",val,"sign:",sign,"location:",location,"csd:",csd,"int:",int_val)
                recovered_wtensor[i_comp, i_col_and_macro] = int_val
        return recovered_wtensor