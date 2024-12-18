import json

import numpy as np

from utils.df_layout import tensor_int8_to_bits
from utils.logger import get_logger

logger = get_logger(__name__)

class MaskConfig:
    def __init__(self, n_from=128, n_to=16):
        self.n_from = n_from
        self.n_to = n_to
        logger.info(f"Mask config: {n_from=}, {n_to=}")

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        n_from = config["mask"]["n_from"]
        n_to = config["mask"]["n_to"]
        return cls(n_from, n_to)


class MaskUtil:
    def __init__(self, mask_memory, macro_config, mask_config):
        self.mask_memory = mask_memory
        self.macro_config = macro_config
        self.mask_config = mask_config

    def get_mask(self, mask_addr, input_size, group_size):
        assert (group_size * input_size) % 8 == 0, f"{group_size=}, {input_size=}"
        mask_size = group_size * input_size // 8
        mask_bytes = self.mask_memory.read(mask_addr, mask_size)
        mask = np.frombuffer(mask_bytes, dtype=np.int8)
        mask = tensor_int8_to_bits(mask)
        mask = mask.reshape(group_size, input_size)
        mask = mask.astype(bool)
        return mask
