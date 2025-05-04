from enum import Enum
class SpecialReg(Enum):

    # pim special reg
    INPUT_BIT_WIDTH = 0
    OUTPUT_BIT_WIDTH = 1
    WEIGHT_BIT_WIDTH = 2
    GROUP_SIZE = 3
    ACTIVATION_GROUP_NUM = 4
    ACTIVATION_ELEMENT_COL_NUM = 5
    GROUP_INPUT_STEP = 6
    GROUP_INPUT_OFFSET_ADDR = 6
    VALUE_SPARSE_MASK_ADDR = 7
    BIT_SPARSE_META_ADDR = 8

    # simd special reg
    SIMD_INPUT_1_BIT_WIDTH = 16
    SIMD_INPUT_2_BIT_WIDTH = 17
    SIMD_INPUT_3_BIT_WIDTH = 18
    SIMD_INPUT_4_BIT_WIDTH = 19
    SIMD_OUTPUT_BIT_WIDTH = 20
    SPECIAL_REG_SIMD_EXTRA_INPUT_ADDR_1 = 21
    SPECIAL_REG_SIMD_EXTRA_INPUT_ADDR_2 = 22

    # Data type
    # Only use in this functional simulator, not used in pimsim
    DTYPE_MACRO_IS_FLOAT = 30
    DTYPE_SIMD_IS_FLOAT = 31