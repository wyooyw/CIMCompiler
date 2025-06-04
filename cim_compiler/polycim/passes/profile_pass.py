import json
import os
import subprocess

from polycim.passes.base import BreadthFirstPass


def profile(temp_dir, pimsim_cfg_path, op_name, op_id):
    # 1. convert format
    op_dir = os.path.join(temp_dir, op_name, op_id)

    cimflow_code_path = os.path.join(op_dir, "final_code.json")
    legacy_code_path = os.path.join(op_dir, "final_code.legacy.json")
    report_path = os.path.join(op_dir, f"pimsim_report.json")

    subprocess.run(
        [
            "cim-compiler",
            "convert",
            "--src-type",
            "cimflow",
            "--dst-type",
            "legacy",
            "--src-file",
            cimflow_code_path,
            "--dst-file",
            legacy_code_path,
            "--filter-out-invalid-instructions",
        ],
        check=True,
    )

    # 2. profile
    """
    pimsim ./pimsim_configs/config-m1g1c32b64.json C1.json -r -c -j ./C1.report.json
    """
    subprocess.run(
        [
            "pimsim",
            pimsim_cfg_path,
            legacy_code_path,
            "-r",
            "-c",
            "-j",
            report_path,
        ],
        check=True,
    )

    # 3. parse report
    with open(report_path, "r") as f:
        report = json.load(f)
    return report


class ProfilePass(BreadthFirstPass):
    def __init__(self, args):
        super().__init__()
        self.op_list = list()
        self.args = args

    def apply(self, operator):
        self.op_list.append(operator)

    def apply_all(self):
        for i, op in enumerate(self.op_list):
            # op_dir = os.path.join(self.args.output_path, op.attr["name"], str(i))
            report = profile(
                self.args.output_path,
                self.args.pimsim_cfg_path,
                op.attr["name"],
                str(i),
            )
            op.attr["ProfilePass"] = {
                "latency": report["latency_"],
                "total_energy": report["total_energy_"],
            }

    def get_result(self):
        return self.op_list
