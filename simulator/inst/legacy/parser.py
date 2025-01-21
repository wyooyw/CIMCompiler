import json
from simulator.inst.instruction import *
from simulator.inst.legacy.dumper import LegacyDumper
from simulator.inst.asm.dumper import AsmDumper
from simulator.inst.asm.parser import AsmParser
class LegacyParser:
    def __init__(self):
        pass

    def parse_file(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data, self.parse(data)

    def parse(self, data):
        
        instructions = []
        for item in data:
            inst = self._parse_inst(item)
            instructions.append(inst)
        return instructions
    
    def _parse_inst(self, inst):
        class_ = inst.get("class", None)
        if class_ == 0b01:
            return self._parse_simd_class_inst(inst)
        elif class_ == 0b10:
            return self._parse_scalar_class_inst(inst)
        elif class_ == 0b110:
            return self._parse_trans_class_inst(inst)
        elif class_ == 0b111:
            return self._parse_control_class_inst(inst)
        elif class_ == 0b00:
            return self._parse_cim_class_inst(inst)
        else:
            raise ValueError(f"Unknown instruction class: {class_}")
    
    def _parse_simd_class_inst(self, inst):
        return SIMDInst(
            opcode=inst["opcode"],
            input_num=inst["input_num"] + 1,
            reg_in1=inst["rs1"],
            reg_in2=inst["rs2"],
            reg_size=inst["rs3"],
            reg_out=inst["rd"]
        )

    def _parse_scalar_class_inst(self, inst):
        type_ = inst.get("type", None)
        opcode = inst.get("opcode", None)

        if type_ == 0b11:
            if opcode == 0b00:
                return GeneralLiInst(
                    reg=inst["rd"],
                    value=inst["imm"]
                )
            elif opcode == 0b01:
                return SpecialLiInst(
                    reg=inst["rd"],
                    value=inst["imm"]
                )
            elif opcode == 0b10:
                return GeneralToSpecialAssignInst(
                    reg_general=inst["rs1"],
                    reg_special=inst["rs2"]
                )
            elif opcode == 0b11:
                return SpecialToGeneralAssignInst(
                    reg_general=inst["rs1"],
                    reg_special=inst["rs2"]
                )
        elif type_ == 0b00:
            return ArithInst(
                opcode=opcode,
                reg_lhs=inst["rs1"],
                reg_rhs=inst["rs2"],
                reg_out=inst["rd"]
            )
        elif type_ == 0b01:
            return RIInst(
                opcode=opcode,
                reg_in=inst["rs"],
                reg_out=inst["rd"],
                imm=inst["imm"]
            )
        elif type_ == 0b10:
            if opcode == 0b00:
                return LoadInst(
                    reg_addr=inst["rs1"],
                    reg_value=inst["rs2"],
                    offset=inst["offset"]
                )
            elif opcode == 0b01:
                return StoreInst(
                    reg_addr=inst["rs1"],
                    reg_value=inst["rs2"],
                    offset=inst["offset"]
                )
        else:
            raise ValueError(f"Unknown scalar instruction type: {type_}")

    def _parse_trans_class_inst(self, inst):
        return TransInst(
            reg_in=inst["rs1"],
            reg_out=inst["rd"],
            reg_size=inst["rs2"],
            flag_src_offset=inst["source_offset_mask"],
            flag_dst_offset=inst["destination_offset_mask"],
            offset=inst["offset"]
        )

    def _parse_control_class_inst(self, inst):
        type_ = inst.get("type", None)

        if type_ == 0b100:
            return JumpInst(
                offset=inst["offset"]
            )
        elif type_ in [0b000, 0b001, 0b010, 0b011]:
            return BranchInst(
                compare=type_,
                reg_lhs=inst["rs1"],
                reg_rhs=inst["rs2"],
                offset=inst["offset"]
            )
        else:
            raise ValueError(f"Unknown control instruction type: {type_}")

    def _parse_cim_class_inst(self, inst):
        type_ = inst.get("type", None)
        if type_ == 0b0:
            return CIMComputeInst(
                reg_input_addr=inst["rs1"],
                reg_input_size=inst["rs2"],
                reg_activate_row=inst["rs3"],
                flag_accumulate=inst["accumulate"],
                flag_value_sparse=inst["value_sparse"],
                flag_bit_sparse=inst["bit_sparse"],
                flag_group=inst["group"],
                flag_group_input_mode=inst["group_input_mode"]
            )
        elif type_ == 0b01:
            return CIMConfigInst(
                reg_single_group_id=inst["rs1"],
                reg_mask_addr=inst["rs2"],
                flag_group_broadcast=inst["group_broadcast"]
            )
        elif type_ == 0b10:
            return CIMOutputInst(
                reg_out_n=inst["rs1"],
                reg_out_mask_addr=inst["rs2"],
                reg_out_addr=inst["rd"],
                flag_outsum=inst["outsum"],
                flag_outsum_move=inst["outsum_move"]
            )
        else:
            raise ValueError(f"Unknown CIM instruction type: {type_}")

if __name__ == "__main__":
    parser = LegacyParser()
    with open("/home/wangyiou/project/cim_compiler_frontend/playground/op/depthwise_conv/.result/final_code.json", 'r') as file:
        data = json.load(file)
    instructions = parser.parse(data)
    # dumper = LegacyDumper(instructions)
    # new_data = dumper.dump()
    # print(json.dumps(data[:4], indent=4))
    # print("-------")
    # print(json.dumps(new_data[:4], indent=4))
    # print(data==new_data)
    asm_dumper = AsmDumper()
    asm_list = asm_dumper.dump(instructions)

    asm_parser = AsmParser()
    instructions2 = asm_parser.parse(asm_list)

    for idx, (inst1, inst2) in enumerate(zip(instructions, instructions2)):
        if inst1 != inst2:
            # inst1 and inst2 are dataclasses
            print(f"{idx} Different: \n\t{inst1=}\n\t{inst2=}")
            for field in inst1.__dataclass_fields__:
                value1 = getattr(inst1, field)
                value2 = getattr(inst2, field)
                print(f"Field '{field}': inst1 type={type(value1)}, inst2 type={type(value2)}")