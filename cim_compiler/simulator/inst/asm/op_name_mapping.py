_sc_funct_name_pair = (
    (0b000000, "SC_ADD"),
    (0b000001, "SC_SUB"),
    (0b000010, "SC_MUL"),
    (0b000011, "SC_DIV"),
    (0b000100, "SC_SLL"),
    (0b000101, "SC_SRL"),
    (0b000110, "SC_SRA"),
    (0b000111, "SC_MOD"),
    (0b001000, "SC_MIN"),
    (0b001001, "SC_MAX"),
    (0b001010, "SC_AND"),
    (0b001011, "SC_OR"),
    (0b001100, "SC_EQ"),
    (0b001101, "SC_NE"),
    (0b001110, "SC_GT"),
    (0b001111, "SC_LT"),
)

_simd_funct_name_pair = (
    (0, "VEC_ADD"),
    (1, "VEC_SC_ADD"),
    (2, "VEC_MUL"),
    (3, "VEC_QUANTIZE"),
)

_branch_compare_name_pair = (
    (0b00, "BEQ"),
    (0b01, "BNE"),
    (0b10, "BGT"),
    (0b11, "BLT"),
)

_special_reg_name_pair = (
    (0, "CIM_IBW"), # Specifies the bit width of input data.
    (1, "CIM_OBW"), # Specifies the bit width of output data.
    (2, "CIM_WBW"), # Specifies the bit width of weights.
    (3, "CIM_GSZ"), # Defines the size of Macro Groups, indicating the number of Macros per Group. Must adhere to values specified in the configuration file.
    (4, "CIM_AG"), # Number of active Groups in the CIM unit.
    (5, "CIM_AE"), # Number of active Elements per column in each Group.
    (6, "CIM_GSTEP"), # Offset address for each group's input vector, defined as the step size relative to the previous group or an offset from the address in register rs1.
    (7, "CIM_VMASK"), # Starting address for the sparse mask for value-level sparsity.
    (8, "CIM_BMETA"), # Starting address for bit-level sparse metadata.
    *((i, f"RESERVED_{i}") for i in range(9, 16)), # Reserved for future extensions.
    # 16-23
    (16, "VEC_IBW1"), # Bit width for each element of Input Vector 1.
    (17, "VEC_IBW2"), # Bit width for each element of Input Vector 2.
    (18, "VEC_IBW3"), # Bit width for each element of Input Vector 3.
    (19, "VEC_IBW4"), # Bit width for each element of Input Vector 4.
    (20, "VEC_OBW"), # Bit width for each element of the output vector.
    (21, "VEC_IA3"), # Starting address for Input Vector 3.
    (22, "VEC_IA4"), # Starting address for Input Vector 4.
    *((i, f"RESERVED_{i}") for i in range(23, 32)), # Reserved for additional instructions or future extensions.
)

# Create mappings for SC functions
mapping_sc_funct_to_name = {code: name for code, name in _sc_funct_name_pair}
mapping_name_to_sc_funct = {name: code for code, name in _sc_funct_name_pair}

# Create mappings for SIMD functions
mapping_simd_funct_to_name = {code: name for code, name in _simd_funct_name_pair}
mapping_name_to_simd_funct = {name: code for code, name in _simd_funct_name_pair}

# Create mappings for branch comparisons
mapping_branch_compare_to_name = {code: name for code, name in _branch_compare_name_pair}
mapping_name_to_branch_compare = {name: code for code, name in _branch_compare_name_pair}

# Create mappings for special registers
mapping_special_reg_to_name = {code: name for code, name in _special_reg_name_pair}
mapping_name_to_special_reg = {name: code for code, name in _special_reg_name_pair}
