import numpy as np
import json

from cim_compiler.utils.logger import get_logger

logger = get_logger(__name__)

class ReduceSumConfig:
    def __init__(self, reduce_len, reduce_num):
        self.reduce_len = reduce_len
        self.reduce_num = reduce_num
        logger.debug(f"Mask config: {reduce_len=}, {reduce_num=}")

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(config.get("reduce_sum", {}).get("reduce_len", None), config.get("reduce_sum", {}).get("reduce_num", None))

class ReduceSumUtil:
    def __init__(self, reduce_sum_config):
        self.reduce_sum_config = reduce_sum_config
        if reduce_sum_config is not None:
            assert isinstance(reduce_sum_config, ReduceSumConfig)
            self.reduce_len = reduce_sum_config.reduce_len
            self.reduce_num = reduce_sum_config.reduce_num
        else:
            self.reduce_len = None
            self.reduce_num = None

    def reduce_sum(self, src_vector):
        if self.reduce_sum_config is None or self.reduce_len is None or self.reduce_num is None:
            assert False, "Reduce sum config is not set"
        assert len(src_vector.shape) == 1
        assert src_vector.shape[0] <= self.reduce_len * self.reduce_num
        # pad src_vector to the nearest multiple of reduce_len
        N = src_vector.shape[0]
        pad_len = (self.reduce_len - N % self.reduce_len) % self.reduce_len
        src_vector = np.pad(src_vector, (0, pad_len), mode='constant')
        N = src_vector.shape[0]
        src_vector = src_vector.reshape(N // self.reduce_len, self.reduce_len)
        dst_vector = src_vector.sum(axis=1)
        dst_vector = dst_vector.reshape(-1)
        return dst_vector


class ReduceMaxConfig:
    def __init__(self, reduce_len, reduce_num):
        self.reduce_len = reduce_len
        self.reduce_num = reduce_num
        logger.debug(f"Mask config: {reduce_len=}, {reduce_num=}")

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        return cls(config.get("reduce_max", {}).get("reduce_len", None), config.get("reduce_max", {}).get("reduce_num", None))

class ReduceMaxUtil:
    def __init__(self, reduce_max_config):
        self.reduce_max_config = reduce_max_config
        if reduce_max_config is not None:
            assert isinstance(reduce_max_config, ReduceMaxConfig)
            self.reduce_len = reduce_max_config.reduce_len
            self.reduce_num = reduce_max_config.reduce_num
        else:
            self.reduce_len = None
            self.reduce_num = None

    def reduce_max(self, src_vector):
        if self.reduce_max_config is None or self.reduce_len is None or self.reduce_num is None:
            assert False, "Reduce max config is not set"
        assert len(src_vector.shape) == 1
        assert src_vector.shape[0] <= self.reduce_len * self.reduce_num
        # pad src_vector to the nearest multiple of reduce_len
        N = src_vector.shape[0]
        pad_len = (self.reduce_len - N % self.reduce_len) % self.reduce_len
        src_vector = np.pad(src_vector, (0, pad_len), mode='constant')
        N = src_vector.shape[0]
        src_vector = src_vector.reshape(N // self.reduce_len, self.reduce_len)
        dst_vector = src_vector.max(axis=1)
        dst_vector = dst_vector.reshape(-1)
        return dst_vector
