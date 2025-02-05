import json
from simulator.inst.instruction import *
from simulator.inst.asm.op_name_mapping import (
    mapping_name_to_sc_funct,
    mapping_name_to_simd_funct,
    mapping_name_to_branch_compare,
    mapping_name_to_special_reg
)
import re

def match_asm_args(args_str, term_types):
    """
    根据term_types生成对应的正则表达式pattern来匹配参数
    term_types = ['reg', 'reg', 'reg', 'imm', 'flag'] 等

    Args:
        args_str: 要匹配的参数字符串
        term_types: 参数类型列表

    Returns:
        匹配到的参数列表
    """
    # 定义各种类型的模式
    patterns = {
        'reg': r'r(\d+)',
        'imm': r'(-?\d+)',
        'flag': r'(?:\s*,\s*(\w+(?:\s*,\s*\w+)*))?',
        'offset_addr': r'(-?\d+\(r\d+\))',
        'sreg': r'([0-9A-Z_]+)'  # 用于特殊寄存器名称
    }
    
    # 构建完整的pattern
    pattern_parts = []
    for t in term_types:
        if t != "flag":
            pattern_parts.append(patterns[t])
    
    # 用逗号和可选的空格连接各部分
    full_pattern = r'\s*' + r'\s*,\s*'.join(pattern_parts)

    if "flag" in term_types:
        full_pattern = full_pattern + patterns["flag"]

    # 编译并匹配
    match = re.match(full_pattern, args_str)

    if not match:
        import pdb; pdb.set_trace()
        raise ValueError(f"Failed to parse arguments: {args_str}")

    match_list = []
    for term_type, match_item in zip(term_types, match.groups()):
        if term_type == 'reg':
            match_list.append(int(match_item))
        elif term_type == 'imm':
            match_list.append(int(match_item))
        elif term_type == 'flag':
            if match_item is None:
                flags = []
            else:
                flags = re.split(r'[(),]', match_item)
                flags = [arg.strip() for arg in flags if arg.strip()]
            match_list.append(flags)
        elif term_type == 'sreg':
            match_list.append(mapping_name_to_special_reg[match_item])
        elif term_type == 'offset_addr':
            addr_pattern = r'(-?\d+)\(r(\d+)\)'
            match = re.match(addr_pattern, match_item)
            if match:
                offset, reg_addr = map(int, match.groups())
                match_list.append((offset, reg_addr))
            else:
                raise ValueError(f"Invalid offset address: {match_item}")
        else:
            raise ValueError(f"Invalid term type: {term_type}")
    return match_list

class AsmParser:
    def __init__(self):
        pass

    def parse_file(self, file_path):
        with open(file_path, 'r') as file:
            data = [line.strip() for line in file.readlines()]
        return data, self.parse(data)

    def parse(self, data):
        instructions = []
        for line in data:
            line = line.strip()
            if line:  # Skip empty lines
                inst = self._parse_inst(line)
                instructions.append(inst)
        return instructions

    def _parse_inst(self, inst_str):
        # Split the instruction into op_name and args
        parts = inst_str.split(maxsplit=1)
        op_name = parts[0]
        args = parts[1] if len(parts) > 1 else ""
        # str_args = args

        # # Split args by comma or parentheses and strip spaces
        # args = re.split(r'[(),]', args)
        # args = [arg.strip() for arg in args if arg.strip()]

        if op_name == "G_LI":
            reg, value = match_asm_args(args, ['reg', 'imm'])
            return GeneralLiInst(reg=reg, value=value)
        elif op_name == "S_LI":
            reg, value = match_asm_args(args, ['sreg', 'imm'])
            return SpecialLiInst(reg=reg, value=value)
        elif op_name == "GS_MOV":
            reg_general, reg_special = match_asm_args(args, ['reg', 'sreg'])
            return GeneralToSpecialAssignInst(reg_special=reg_special, reg_general=reg_general)
        elif op_name == "SG_MOV":
            reg_special, reg_general = match_asm_args(args, ['sreg', 'reg'])
            return SpecialToGeneralAssignInst(reg_general=reg_general, reg_special=reg_special)
        elif op_name == "SC_LD":
            reg_value, (offset, reg_addr) = match_asm_args(args, ['reg', 'offset_addr'])
            return LoadInst(reg_addr=reg_addr, reg_value=reg_value, offset=offset)
        elif op_name == "SC_ST":
            reg_value, (offset, reg_addr) = match_asm_args(args, ['reg', 'offset_addr'])
            return StoreInst(reg_addr=reg_addr, reg_value=reg_value, offset=offset)
        elif op_name.startswith("SC_") and not op_name.endswith("I"):
            reg_out, reg_lhs, reg_rhs = match_asm_args(args, ['reg', 'reg', 'reg'])
            opcode = self._name_to_arith_funct(op_name)
            return ArithInst(opcode=opcode, reg_lhs=reg_lhs, reg_rhs=reg_rhs, reg_out=reg_out)
        elif op_name.startswith("SC_") and op_name.endswith("I"):
            reg_out, reg_in, imm = match_asm_args(args, ['reg', 'reg', 'imm'])
            opcode = self._name_to_arith_funct(op_name[:-1], is_ri=True)
            return RIInst(opcode=opcode, reg_in=reg_in, reg_out=reg_out, imm=imm)
        elif op_name.startswith("VEC_"):
            reg_out, reg_in1, reg_in2, reg_size, input_num = match_asm_args(args, ['reg', 'reg', 'reg', 'reg', 'imm'])
            opcode = self._name_to_simd_funct(op_name)
            return SIMDInst(opcode=opcode, reg_in1=reg_in1, reg_in2=reg_in2, reg_size=reg_size, reg_out=reg_out, input_num=input_num)
        elif op_name == "MEM_CPY":
            reg_out, reg_in, reg_size, offset, flags = match_asm_args(args, ['reg', 'reg', 'reg', 'imm', 'flag'])
            return TransInst(
                reg_out=reg_out,
                reg_in=reg_in,
                reg_size=reg_size,
                flag_src_offset=int("SRC_O" in flags),
                flag_dst_offset=int("DST_O" in flags),
                offset=offset
            )

        # elif op_name == "SC_LD":
        #     reg_value, offset_reg_addr = args
        #     reg_value = int(reg_value)
        #     offset, reg_addr = map(int, offset_reg_addr.strip('()').split(','))
        #     return LoadInst(reg_addr=reg_addr, reg_value=reg_value, offset=offset)
        # elif op_name == "SC_ST":
        #     reg_value, offset_reg_addr = args
        #     reg_value = int(reg_value)
        #     offset, reg_addr = map(int, offset_reg_addr.strip('()').split(','))
        #     return StoreInst(reg_addr=reg_addr, reg_value=reg_value, offset=offset)
        elif op_name == "PRINT":
            reg = match_asm_args(args, ['reg'])
            return PrintInst(reg=reg)
        elif op_name == "DEBUG":
            return DebugInst()
        elif op_name in ["BEQ", "BNE", "BGT", "BLT"]:
            reg_lhs, reg_rhs, offset = match_asm_args(args, ['reg', 'reg', 'imm'])
            compare = self._name_to_branch_compare(op_name)
            return BranchInst(compare=compare, reg_lhs=reg_lhs, reg_rhs=reg_rhs, offset=offset)
        elif op_name == "JMP":
            offset, = match_asm_args(args, ['imm'])
            return JumpInst(offset=offset)
        
        elif op_name == "CIM_MVM":
            
            # CIM_MVM r14, r11, r10, GRP
            reg_input_addr, reg_input_size, reg_activate_row, flags = match_asm_args(args, ['reg', 'reg', 'reg', 'flag'])

            flag_value_sparse = "SP_V" in flags
            flag_bit_sparse = "SP_B" in flags
            flag_group = "GRP" in flags
            flag_group_input_mode = "GRP_I" in flags

            return CIMComputeInst(
                reg_input_addr=int(reg_input_addr),
                reg_input_size=int(reg_input_size), 
                reg_activate_row=int(reg_activate_row), 
                flag_value_sparse=int(flag_value_sparse), 
                flag_bit_sparse=int(flag_bit_sparse), 
                flag_group=int(flag_group), 
                flag_group_input_mode=int(flag_group_input_mode),
                flag_accumulate = 1
            )

        elif op_name == "CIM_CFG":
            
            reg_single_group_id, reg_mask_addr, flags = match_asm_args(args, ['reg', 'reg', 'flag'])
            
            return CIMConfigInst(
                reg_single_group_id=reg_single_group_id,
                reg_mask_addr=reg_mask_addr,
                flag_group_broadcast=int("GRP_B" in flags)
            )

        elif op_name == "CIM_OUT":

            reg_out_n, reg_out_mask_addr, reg_out_addr, flags = match_asm_args(args, ['reg', 'reg', 'reg', 'flag'])
            
            return CIMOutputInst(
                reg_out_n=reg_out_n,
                reg_out_mask_addr=reg_out_mask_addr,
                reg_out_addr=reg_out_addr,
                flag_outsum=int("OSUM" in flags),
                flag_outsum_move=int("OSUM_MOVE" in flags)
            )
        else:
            raise ValueError(f"Unknown instruction name: {op_name}")

    def _name_to_arith_funct(self, name, is_ri=False):
        op_name = mapping_name_to_sc_funct[name]
        return op_name

    def _name_to_simd_funct(self, name):
        op_name = mapping_name_to_simd_funct[name]
        return op_name

    def _name_to_branch_compare(self, name):
        op_name = mapping_name_to_branch_compare[name]
        return op_name
        