import os
import tempfile
import subprocess
import numpy as np
from jinja2 import Environment, FileSystemLoader, StrictUndefined
from dataclasses import dataclass
import copy
from multiprocessing import Process


class OpRunner:
    def __init__(self, op_path, op_config, cim_config_path):
        self.op_path = op_path
        self.op_config = op_config
        self.cim_config_path = cim_config_path

    def run(self, input_list:list[np.ndarray], output_list:list[np.ndarray]):
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # tmp_dir = "/home/wangyiou/project/CIMCompiler/.temp"
            op_code_path = os.path.join(tmp_dir, "op_code.cim")
            final_code_dir = os.path.join(tmp_dir, "compiler_output")
            simulator_output_dir = os.path.join(tmp_dir, "simulator_output")
            image_path = os.path.join(tmp_dir, "image.bin")
            self.fill_template(self.op_path, self.op_config, op_code_path)
            self.compile(op_code_path, final_code_dir)
            self.make_image(input_list, image_path)
            self.simulate(image_path, final_code_dir, simulator_output_dir)
            self.get_output(simulator_output_dir, input_list, output_list)

    def get_output(self, simulator_output_dir:str, input_list:list[np.ndarray], output_list:list[np.ndarray]):
        # Calculate total size of input arrays in bytes
        input_size = sum(arr.nbytes for arr in input_list)
        
        # Read the output data from image.bin
        image_path = os.path.join(simulator_output_dir, "image.bin")
        with open(image_path, 'rb') as f:
            # Skip input data
            f.seek(input_size)
            
            # Read output data for each output array
            for output_arr in output_list:
                output_bytes = f.read(output_arr.nbytes)
                # Copy bytes directly into the output array
                output = np.frombuffer(output_bytes, dtype=output_arr.dtype).reshape(output_arr.shape)
                output_arr[:] = output

    def simulate(self, image_path:str, final_code_dir:str, simulator_output_dir:str):
        subprocess.run([
            "cim-compiler", "simulate",
            "--code-file", os.path.join(final_code_dir, "final_code.json"),
            "--data-file", image_path,
            "--config-file", self.cim_config_path,
            "--output-dir", simulator_output_dir,
            "--code-format", "cimflow",
            "--save-stats"
        ], check=True)

    def make_image(self, input_list:list[np.ndarray], image_path:str):
        # 将所有输入张量转换为字节数组并拼接
        image_byte_array = bytearray()
        for tensor in input_list:
            tensor_byte_array = bytearray(tensor)
            image_byte_array += tensor_byte_array
        
        # 将拼接后的字节数组写入文件
        with open(image_path, 'wb') as f:
            f.write(image_byte_array)

    def compile(self, op_code_path:str, final_code_dir:str):
        subprocess.run([
            "cim-compiler", "compile",
            "--input-file", op_code_path,
            "--output-dir", final_code_dir,
            "--config-file", self.cim_config_path
        ], check=True)

    def fill_template(self, src_path:str, context, dst_path:str=None):
        src_folder, src_file = os.path.split(src_path)

        env = Environment(
            loader=FileSystemLoader([
                src_folder, 
                os.environ["CIM_COMPILER_BASE"],
                os.environ.get(os.environ["CIM_COMPILER_BASE"], "cim_compiler")
            ]),
            undefined=StrictUndefined
        )
        template = env.get_template(src_file)
        output = template.render(context.__dict__)

        if dst_path:
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            with open(dst_path, "w") as f:
                f.write(output)
        
        return output

@dataclass
class SIMDOpConfig:
    core_id: int = -1
    world_size: int = -1

class SPMDOpRunner(OpRunner):
    def __init__(self, op_path, op_config, cim_config_path, num_cores:int, config_for_each_core = None):
        super().__init__(op_path, op_config, cim_config_path)
        assert isinstance(op_config, SIMDOpConfig)
        self.num_cores = num_cores
        self.config_for_each_core = config_for_each_core

    def _compile_core(self, core_id, tmp_dir, input_list):
        op_config = copy.deepcopy(self.op_config)
        op_config.core_id = core_id
        op_config.world_size = self.num_cores
        if self.config_for_each_core:
            self.config_for_each_core(core_id, op_config)
        
        core_tmp_dir = os.path.join(tmp_dir, str(core_id))
        op_code_path = os.path.join(core_tmp_dir, "op_code.cim")
        final_code_dir = os.path.join(core_tmp_dir, "compiler_output")
        image_path = os.path.join(core_tmp_dir, "image.bin")
        self.fill_template(self.op_path, op_config, op_code_path)
        self.compile(op_code_path, final_code_dir)
        self.make_image(input_list, image_path)

    def run(self, input_list:list[list[np.ndarray]], output_list:list[list[np.ndarray]]):
        assert len(input_list) == len(output_list) == self.num_cores, f"{len(input_list)=}, {len(output_list)=}, {self.num_cores=}"
        assert all(len(input_list[i]) == len(input_list[0]) for i in range(self.num_cores))
        assert all(len(output_list[i]) == len(output_list[0]) for i in range(self.num_cores))

        with tempfile.TemporaryDirectory() as tmp_dir:
            # tmp_dir = "/home/wangyiou/project/CIMCompiler/.temp"
            
            processes = []
            for core_id in range(self.num_cores):
                p = Process(target=self._compile_core, args=(core_id, tmp_dir, input_list[core_id]))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
            
            core_tmp_dir = os.path.join(tmp_dir, "{core_id}")
            final_code_dir = os.path.join(core_tmp_dir, "compiler_output")
            simulator_output_dir = os.path.join(core_tmp_dir, "simulator_output")
            image_path = os.path.join(core_tmp_dir, "image.bin")
            self.simulate(image_path, final_code_dir, simulator_output_dir)

            for core_id in range(self.num_cores):
                core_tmp_dir = os.path.join(tmp_dir, str(core_id))
                simulator_output_dir = os.path.join(core_tmp_dir, "simulator_output")
                self.get_output(simulator_output_dir, input_list[core_id], output_list[core_id])

    def simulate(self, image_path:str, final_code_dir:str, simulator_output_dir:str):
        subprocess.run([
            "cim-compiler", "multi-core-simulate",
            "--code-file", os.path.join(final_code_dir, "final_code.json"),
            "--data-file", image_path,
            "--config-file", self.cim_config_path,
            "--output-dir", simulator_output_dir,
            "--code-format", "cimflow",
            "--save-stats",
            "--num-cores", str(self.num_cores)
        ], check=True)