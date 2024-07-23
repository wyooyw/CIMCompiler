import numpy as np
class MaskConfig:
    def __init__(self, n_from=128, n_to=16):
        self.n_from = n_from
        self.n_to = n_to

class MaskUtil:
    def __init__(self, mask_memory, macro_config, mask_config):
        self.mask_memory = mask_memory
        self.macro_config = macro_config
        self.mask_config = mask_config

    def get_mask(self, mask_addr, input_size, group_size):
        mask_size = group_size * input_size # TODO: use 1bit as one mask, so the size should div 8.
        mask_bytes = self.mask_memory.read(mask_addr, mask_size)
        mask = np.frombuffer(mask_bytes, dtype=np.int8)
        mask = mask.reshape(group_size, input_size)
        mask = mask.astype(bool)
        return mask