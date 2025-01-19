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

# Create mappings for SC functions
mapping_sc_funct_to_name = {code: name for code, name in _sc_funct_name_pair}
mapping_name_to_sc_funct = {name: code for code, name in _sc_funct_name_pair}

# Create mappings for SIMD functions
mapping_simd_funct_to_name = {code: name for code, name in _simd_funct_name_pair}
mapping_name_to_simd_funct = {name: code for code, name in _simd_funct_name_pair}

# Create mappings for branch comparisons
mapping_branch_compare_to_name = {code: name for code, name in _branch_compare_name_pair}
mapping_name_to_branch_compare = {name: code for code, name in _branch_compare_name_pair}