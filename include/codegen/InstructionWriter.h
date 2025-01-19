// InstructionWriter.h

#ifndef INSTRUCTIONWRITER_H
#define INSTRUCTIONWRITER_H

#include <map>
#include <string>

typedef std::map<std::string, int> Inst;

// Define an interface for InstructionWriter
class InstructionWriter {
public:
    virtual Inst getGeneralLIInst(int reg, int value) = 0;
    virtual Inst getSpecialLIInst(int reg, int value) = 0;
    virtual Inst getSpecialToGeneralAssignInst(int reg_general, int reg_special) = 0;
    virtual Inst getGeneralToSpecialAssignInst(int reg_general, int reg_special) = 0;
    virtual Inst getArithInst(int opcode, int reg_in1, int reg_in2, int reg_out) = 0;
    virtual Inst getRIInst(int opcode, int reg_in, int reg_out, int imm) = 0;
    virtual Inst getSIMDInst(int opcode, int input_num, int in1_reg, int in2_reg, int size_reg, int out_reg) = 0;
    virtual Inst getTransInst(int reg_addr_in, int reg_addr_out, int size) = 0;
    virtual Inst getLoadInst(int reg_addr, int reg_value, int offset) = 0;
    virtual Inst getStoreInst(int reg_addr, int reg_value, int offset) = 0;
    virtual Inst getPrintInst(int reg) = 0;
    virtual Inst getDebugInst() = 0;
    virtual Inst getBranchInst(int compare, int reg1, int reg2, int offset) = 0;
    virtual Inst getJumpInst(int offset) = 0;
    virtual Inst getCIMComputeInst(int reg_input_addr, int reg_input_size, int reg_activate_row, int flag_accumulate, int flag_value_sparse, int flag_bit_sparse, int flag_group, int flag_group_input_mode) = 0;

    virtual void setJumpOffset(Inst &inst, int offset) = 0;
    virtual void setBranchOffset(Inst &inst, int offset) = 0;

    virtual bool isGeneralReg(Inst &inst, std::string key) = 0; 
    virtual bool isGeneralToSpecialAssign(Inst &inst) = 0;
    virtual bool isSpecialToGeneralAssign(Inst &inst) = 0;
    virtual bool isSpecialLi(Inst &inst) = 0;
    virtual ~InstructionWriter() = default;
};

// Implement a concrete InstructionWriter
class LegacyInstructionWriter : public InstructionWriter {
public:
    Inst getGeneralLIInst(int reg, int value) override;
    Inst getSpecialLIInst(int reg, int value) override;
    Inst getSpecialToGeneralAssignInst(int reg_general, int reg_special) override;
    Inst getGeneralToSpecialAssignInst(int reg_general, int reg_special) override;
    Inst getArithInst(int opcode, int reg_in1, int reg_in2, int reg_out) override;
    Inst getRIInst(int opcode, int reg_in, int reg_out, int imm) override;
    Inst getSIMDInst(int opcode, int input_num, int in1_reg, int in2_reg, int size_reg, int out_reg) override;
    Inst getTransInst(int reg_addr_in, int reg_addr_out, int size) override;
    Inst getLoadInst(int reg_addr, int reg_value, int offset) override;
    Inst getStoreInst(int reg_addr, int reg_value, int offset) override;
    Inst getPrintInst(int reg) override;
    Inst getDebugInst() override;
    Inst getJumpInst(int offset) override;
    Inst getBranchInst(int compare, int reg1, int reg2, int offset) override;
    void setJumpOffset(Inst &inst, int offset);
    void setBranchOffset(Inst &inst, int offset);
    Inst getCIMComputeInst(int reg_input_addr, int reg_input_size, int reg_activate_row, int flag_accumulate, int flag_value_sparse, int flag_bit_sparse, int flag_group, int flag_group_input_mode) override;

    bool isGeneralReg(Inst &inst, std::string key) override; 
    bool isGeneralToSpecialAssign(Inst &inst) override;
    bool isSpecialToGeneralAssign(Inst &inst) override;
    bool isSpecialLi(Inst &inst) override;
};

// Another implementation
class CIMFlowInstructionWriter : public InstructionWriter {
public:
    Inst getGeneralLIInst(int reg, int value) override;
    Inst getSpecialLIInst(int reg, int value) override;
    Inst getSpecialToGeneralAssignInst(int reg_general, int reg_special) override;
    Inst getGeneralToSpecialAssignInst(int reg_general, int reg_special) override;
    Inst getArithInst(int opcode, int reg_in1, int reg_in2, int reg_out) override;
    Inst getRIInst(int opcode, int reg_in, int reg_out, int imm) override;
    Inst getSIMDInst(int opcode, int input_num, int in1_reg, int in2_reg, int size_reg, int out_reg) override;
    Inst getTransInst(int reg_addr_in, int reg_addr_out, int size) override;
    Inst getLoadInst(int reg_addr, int reg_value, int offset) override;
    Inst getStoreInst(int reg_addr, int reg_value, int offset) override;
    Inst getPrintInst(int reg) override;
    Inst getDebugInst() override;
    Inst getJumpInst(int offset) override;
    Inst getBranchInst(int compare, int reg1, int reg2, int offset) override;
    void setJumpOffset(Inst &inst, int offset);
    void setBranchOffset(Inst &inst, int offset);
    Inst getCIMComputeInst(int reg_input_addr, int reg_input_size, int reg_activate_row, int flag_accumulate, int flag_value_sparse, int flag_bit_sparse, int flag_group, int flag_group_input_mode) override;
    
    bool isGeneralReg(Inst &inst, std::string key) override; 
    bool isGeneralToSpecialAssign(Inst &inst) override;
    bool isSpecialToGeneralAssign(Inst &inst) override;
    bool isSpecialLi(Inst &inst) override;
};

#endif // INSTRUCTIONWRITER_H