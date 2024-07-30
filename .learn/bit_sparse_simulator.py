import torch
from simulator.simple_simulator import SimulatorExecutor,Simulator,Memory,Macro
from utils.df_layout import tensor_int8_to_bits
from op.bit_sparse.weight_transform import recover_csd, csd_to_int
import numpy as np
import json
from config.cim_config import CimConfig
from config.datatype import DATA_TYPE_BYTES
from functools import reduce
import os
import ipdb
from tqdm import tqdm

class BitSparseSimulator(Simulator):
    def __init__(self,cim_cfg):
        super().__init__(cim_cfg)
        self.info_mem = Memory(cim_cfg.info_size, "local.info")
        self.info_reg = Memory(cim_cfg.info_reg_size, "local.info.reg")

    def _run_setpw(self,code):
        self.macro_ibw = code["ibw"]
        self.macro_wbw = code["wbw"]
        self.macro_obw = code["obw"]
        assert self.macro_ibw in [8,16,32], f"Currently, we only support macro_ibw is divisable by byte, but we got macro_ibw={self.macro_ibw}"
        assert self.macro_wbw in [1,8,16,32], f"Currently, we only support macro_wbw is 1 in bit_sparse mode, but we got macro_wbw={self.macro_wbw}"
        assert self.macro_obw in [8,16,32], f"Currently, we only support macro_obw is divisable by byte, but we got macro_obw={self.macro_obw}"

    def _run_trans(self,code):
        """
        {
            "name": "trans",
            "params": {
                "memd": 0,
                "mems": 1,
                "size": 24,
                "saddr": 0,
                "daddr": 0
            }
        }
        trans $rd $rs1 offd offs memd mems size 
        """
        assert code["op"]=="trans"
        memory = [self.local_mem, self.global_mem, self.info_mem, self.local_weight_mem]
        src_memory = memory[code["mems"]]
        dst_memory = memory[code["memd"]]

        size = code["size"]
        data = src_memory.read(code["offs"], size)
        # data_np = np.frombuffer(data,dtype="int8")
        # if code["memd"]>=2:
        #     ipdb.set_trace()
        dst_memory.write(code["offd"], size, data)

    def _run_pim_mult_csd(self,code):
        """
            "name": "pim-mult-csd",
            "params": {
                "broadcast": 1,
                "ibuf_addr": 272,
                "obuf_addr": 288,
                "macro_idx": 0,
                "n_comp": 16,
                "row_idx": 17,
                "accumulate": 1
            }
        """
        # assert code["broadcast"]==1, f"Currently, we only support broadcast=1, but we got broadcast={code['broadcast']}"
        # Get input vector
        iaddr = code["iaddr_off"]
        isize_bit = self.macro_ibw * code["comp_off"]
        assert self.macro_ibw % 8 == 0, f"Currently, we only support macro.ibw is align by byte, but we got macro_ibw={self.macro_ibw}"
        isize = isize_bit // 8
        idata = self.pim_buffer.read(iaddr,isize)
        itensor = np.frombuffer(idata, dtype=f"int{self.macro_ibw}").reshape(1,-1)

        # Get weight matrix
        # Stack to get whole weight matrix
        macro_num = code["macro_off"] if code["broadcast"]==1 else 1
        macro_begin = 0 if code["broadcast"]==1 else code["macro_off"]
        wdata = bytearray()
        for comp_idx in range(code["comp_off"]):
            waddr = code["row_off"] * (self.cim_cfg.n_compartment * self.cim_cfg.n_macro * self.cim_cfg.bytes_column) + \
                    comp_idx * (self.cim_cfg.n_macro * self.cim_cfg.bytes_column) + \
                    macro_begin * (self.cim_cfg.bytes_column)
            wsize = self.cim_cfg.bytes_column * macro_num
            wdata_part = self.macro.read(waddr,wsize)
            wdata += wdata_part
        wtensor = np.frombuffer(wdata, dtype=f"int8").reshape(code["comp_off"], self.cim_cfg.get_n_column(f"int8")*macro_num)
        # print("wtensor:",wtensor)
        wtensor = tensor_int8_to_bits(wtensor)
        wtensor = wtensor.reshape(code["comp_off"], self.cim_cfg.bits_column * macro_num)
        
        # Get info matrix
        info_begin = code["info_offset"]
        info_size = code["comp_off"] * self.cim_cfg.n_macro * self.cim_cfg.bits_column * 3 // 8
        info_data = self.info_reg.read(info_begin, info_size)
        info_tensor = np.frombuffer(info_data, dtype="int8").reshape(-1)
        # print("info_tensor:",info_tensor)
        # ipdb.set_trace()
        info_tensor = tensor_int8_to_bits(info_tensor)
        info_tensor = info_tensor.reshape(code["comp_off"],self.cim_cfg.n_macro , self.cim_cfg.bits_column , 3)
        info_tensor = info_tensor[:,0:macro_num, :,:]
        info_tensor = info_tensor.reshape(code["comp_off"], self.cim_cfg.bits_column * macro_num, 3)
        

        # Recover weight matrix
        # recover_csd, csd_to_int
        recovered_wtensor = np.zeros((code["comp_off"], self.cim_cfg.bits_column * macro_num), dtype=f"int{self.macro_obw}")
        for i_comp in range(code["comp_off"]):
            for i_col_and_macro in range(self.cim_cfg.bits_column * macro_num):
                val = wtensor[i_comp, i_col_and_macro]
                sign = info_tensor[i_comp, i_col_and_macro, 0]
                location = info_tensor[i_comp, i_col_and_macro, 1:3]
                # print(type(val),type(sign),type(location))
                csd = recover_csd(val, sign, location)
                int_val = csd_to_int(csd)
                # print(f"value:",val,"sign:",sign,"location:",location,"csd:",csd,"int:",int_val)
                recovered_wtensor[i_comp, i_col_and_macro] = int_val
        # print(f"recovered_wtensor:\n",recovered_wtensor)
        # ipdb.set_trace()
        # exit()
        # Accumulate
        aaddr = code["oaddr_off"]
        asize_bit = self.macro_obw * self.cim_cfg.bits_column * macro_num
        asize = asize_bit // 8
        adata = self.pim_buffer.read(aaddr, asize)
        atensor = np.frombuffer(adata, dtype=f"int{self.macro_obw}").reshape(1,-1)

        # Calculate output vector
        otensor = np.matmul(itensor, recovered_wtensor, dtype=f"int{self.macro_obw}")
        # print("===============================")
        # print("input:\n",itensor)
        # print("weight:\n",recovered_wtensor)
        
        # print(f"otensor:\n",otensor)
        # print(f"atensor:\n",atensor)
        
        otensor = otensor + atensor
        # print(f"fotensor:\n",otensor)

        odata = otensor.tobytes()

        # Store the output
        oaddr = aaddr
        osize = asize
        self.pim_buffer.write(oaddr,osize,odata)

    # def _run_pim_load_info(self, code):
    #     """
    #         "name": "pim-load-info",
    #         "params": {
    #             "col_n": 16,
    #             "comp_n": 16,
    #             "macro_n": 2,
    #             "addr": 0
    #         }
    #     """
    #     size = 3 * code["col_n"] * code["comp_n"] * code["macro_n"] // 8
    #     # Currently, for convenience,  info in info_memory is not continuous, so we should move multiple times.
    #     # In the future, I would let info be continous in info_memory. 
        
    #     assert code["col_n"]==self.cim_cfg.bits_column, f"Currently, we only support code['col_n']==cim_cfg.bits_column, but got code['col_n']={code['col_n']} and cim_cfg.bits_column={self.cim_cfg.bits_column}"
    #     data = bytearray()
    #     for comp_i in range(code["comp_n"]):
    #         sub_data = self.info_mem.read(code["info_offset"] + comp_i * (self.cim_cfg.bits_column * self.cim_cfg.n_macro * 3 // 8), 
    #                                         self.cim_cfg.bits_column * code["macro_n"] * 3 // 8)
    #         data+=sub_data
    #     data_np = np.frombuffer(data, dtype="int8")
    #     # ipdb.set_trace()
    #     self.info_reg.write(0, size, data)

    def _run_pim_load_info(self, code):
        """
            "name": "pim-load-info",
            "params": {
                "size": 16,
                "info_offset": 0
            }
        """
        data = self.info_mem.read(code["info_offset"], code["size"])
        self.info_reg.write(0, code["size"], data)

    def _run_pim_load_info_mult_csd(self, code):
        self._run_pim_load_info(code)
        self._run_pim_mult_csd(code)

    def _run_pim_outsum(self, code):
        """
            "name": "pim-outsum",
            "params": {
                "out_mask": "01010101001001000000010010101010",
                "out_n": 32,
                "oaddr": 288
            }
        """
        assert self.macro_obw >= 8 and self.macro_obw % 8 == 0
        i = 0
        # print(code["out_mask"])
        # ls = []
        # while i < code["out_n"]:
        #     data1 = self.pim_buffer.read(code["oaddr"] + i * self.macro_obw // 8, self.macro_obw // 8)
        #     tensor1 = np.frombuffer(data1, dtype=f"int{self.macro_obw}").reshape(1,)
        #     ls.append(tensor1.tolist())
        #     i += 1
        # print(ls)
        i = 0
        while i < code["out_n"]:
            if code["out_mask"][i]=="1":
                assert i+1 < len(code["out_mask"])
                assert code["out_mask"][i+1]=="0"
                data1 = self.pim_buffer.read(code["oaddr_off"] + i * self.macro_obw // 8, self.macro_obw // 8)
                data2 = self.pim_buffer.read(code["oaddr_off"] + (i+1) * self.macro_obw // 8, self.macro_obw // 8)
                tensor1 = np.frombuffer(data1, dtype=f"int{self.macro_obw}").reshape(1,)
                tensor2 = np.frombuffer(data2, dtype=f"int{self.macro_obw}").reshape(1,)
                tensor_new = tensor1 + tensor2
                data_new = tensor_new.tobytes()
                self.pim_buffer.write(code["oaddr_off"] + i * self.macro_obw // 8, self.macro_obw // 8, data_new)
            else:
                data1 = self.pim_buffer.read(code["oaddr_off"] + i * self.macro_obw // 8, self.macro_obw // 8)
                tensor1 = np.frombuffer(data1, dtype=f"int{self.macro_obw}").reshape(1,)
            i += 1

    def _run_pim_outsum_move(self, code):
        """
            "name": "pim-outsum-move",
            "params": {
                "out_n": 32,
                "oaddr_off": 288
            }
        """
        assert self.macro_obw >= 8 and self.macro_obw % 8 == 0
        for i in range(code["out_n"]):
            data1 = self.pim_buffer.read(code["oaddr_off"] + 2 * i * self.macro_obw // 8, self.macro_obw // 8)
            data2 = self.pim_buffer.read(code["oaddr_off"] + (2 * i + 1) * self.macro_obw // 8, self.macro_obw // 8)
            tensor1 = np.frombuffer(data1, dtype=f"int{self.macro_obw}").reshape(1,)
            tensor2 = np.frombuffer(data2, dtype=f"int{self.macro_obw}").reshape(1,)
            tensor_new = tensor1 + tensor2
            data_new = tensor_new.tobytes()
            self.pim_buffer.write(code["oaddr_off"] + i * self.macro_obw // 8, self.macro_obw // 8, data_new)

    def run_code(self, code):
        if code["op"]=="setpw":
            self._run_setpw(code)
        elif code["op"]=="pim_mult_csd":
            self._run_pim_mult_csd(code)
        elif code["op"]=="pim_load_info":
            self._run_pim_load_info(code)
        elif code["op"]=="pim_outsum":
            self._run_pim_outsum(code)
        elif code["op"]=="pim_outsum_move":
            self._run_pim_outsum_move(code)
        elif code["op"]=="pim_load_info_mult_csd":
            self._run_pim_load_info_mult_csd(code)
        else:
            super().run_code(code)
        
    def run(self,code_list):
        for code in tqdm(code_list):
            self.run_code(code)

class BitSparseSimulatorExecutor(SimulatorExecutor):
    def __init__(self, cim_cfg, program, const):
        super().__init__( cim_cfg, program, const)
        self.simulator = BitSparseSimulator(cim_cfg)


if __name__=="__main__":
    # Prepare data
    path = "SubGraph/simple/3"
    # Layer info
    with open(os.path.join("model",path,"layer.json"),"r") as f:
        layer = json.load(f)
    # Const
    with open(os.path.join("save",path,"packed_const"),"rb") as f:
        weight_bytes = f.read()
        weight_bytes = bytearray(weight_bytes)
    # Prepare program
    with open(os.path.join("save",path,"program.json"),"r") as f:
        program = json.load(f)
        if "type" in layer and layer["type"]=="FCN":
            in_shape = program["input"]["shape"]
            out_shape = program["output"]["shape"]
        else:
            b,h,w,c = program["input"]["shape"]
            in_shape = [b,c,h,w]
            b,h,w,c = program["output"]["shape"]
            out_shape = [b,c,h,w]
    input = np.load(os.path.join("model",path,"input.npy")).reshape(in_shape).astype("int8")
    print(input.max(), input.min())
    
    if not ("type" in layer and layer["type"]=="FCN"):
        input = np.transpose(input,(0,2,3,1))
    output = np.load(os.path.join("model",path,"output.npy")).reshape(out_shape).astype("int8")
    # 
    
    # Execute
    executor = BitSparseSimulatorExecutor(cim_cfg=CimConfig.global_cim_config(), program=program, const=weight_bytes)
    run_output = executor.run(input)
    if not ("type" in layer and layer["type"]=="FCN"):
        run_output = np.transpose(run_output,(0,3,1,2))
    print("output:\n",output.reshape(output.shape[1],-1))
    print("run_output:\n",run_output.reshape(output.shape[1],-1))
    

    # Check
    print((output==run_output).reshape(-1))
    print("Shape equals: ",output.shape==run_output.shape)
    print("Dtype equals: ",output.dtype==run_output.dtype)
    print("Element equals: ",(output==run_output).all())