import json
import os
from collections import defaultdict
from enum import Enum

from utils.logger import get_logger

logger = get_logger(__name__)

class InstClass(Enum):
    PIM_CLASS = 0  # 0b00
    SIMD_CLASS = 1  # 0b01
    SCALAR_CLASS = 2  # 0b10
    TRANS_CLASS = 6  # 0b110
    CTR_CLASS = 7  # 0b111
    DEBUG_CLASS = -1


class StatsUtil:
    def __init__(self):
        self.total_inst_cnt = 0

        self.per_class_cnt = defaultdict(lambda: defaultdict(int))

        self._last_pim_compute = None
        self._pim_compute_duration = []

        self._trans_addr_max = 0
        self._trans_addr_plus_size_max = 0
        self._reg_data = []

        self._pimset_mask = []
        self._weight_bit_width = None
        self._col_use = 0
        self._col_total = 0

        self._macro_comp_use = []
        self._macro_col_use = []
        self._macro_cell_total = []

    def record_pimset_mask(self, mask, weight_bit_width):
        self._pimset_mask.append(mask)
        self._weight_bit_width = weight_bit_width

    def record_trans_addr(self, src_addr, dst_addr, size):
        self._trans_addr_max = max(self._trans_addr_max, src_addr, dst_addr)
        self._trans_addr_plus_size_max = max(
            self._trans_addr_plus_size_max, src_addr + size, dst_addr + size
        )

    def record_reg_status(self, pc, idx, general_rf):
        if len(self._reg_data) > 10000:
            return
        self._reg_data.append(
            {"pc": pc, "idx": idx, "general_rf": general_rf[:32].tolist()}
        )

    def record(self, inst):
        inst_class = inst["class"]
        if inst_class == InstClass.PIM_CLASS.value:
            self._record_pim_class_inst(inst)
        elif inst_class == InstClass.SIMD_CLASS.value:
            self._record_simd_class_inst(inst)
        elif inst_class == InstClass.SCALAR_CLASS.value:
            self._record_scalar_class_inst(inst)
        elif inst_class == InstClass.TRANS_CLASS.value:
            self._record_trans_class_inst(inst)
        elif inst_class == InstClass.CTR_CLASS.value:
            self._record_control_class_inst(inst)
        else:
            self._record_other_class_inst(inst)

        self.total_inst_cnt += 1

    def _record_pim_compute_duration(self):
        if self._last_pim_compute is None:
            self._last_pim_compute = self.total_inst_cnt
        else:
            self._pim_compute_duration.append(
                self.total_inst_cnt - self._last_pim_compute - 1
            )
            self._last_pim_compute = self.total_inst_cnt

    def record_macro_ultilize(self, use_comp, use_col, total):
        assert use_comp * use_col <= total, f"{use_comp=}, {use_col=}, {total=}"
        self._macro_comp_use.append(use_comp)
        self._macro_col_use.append(use_col)
        self._macro_cell_total.append(total)

    def _record_use_col(self):
        pimset_mask = self._pimset_mask[-1]
        col_use = sum(pimset_mask) * self._weight_bit_width
        col_total = len(pimset_mask) * self._weight_bit_width
        assert col_total == 128, f"{col_total=}"
        assert col_use <= col_total, f"{col_use=}, {col_total=}"

        self._col_use += col_use
        self._col_total += col_total

    def _record_pim_class_inst(self, inst):
        self.per_class_cnt["pim"]["total"] += 1
        if inst["type"] == 0b00:
            self.per_class_cnt["pim"]["pim_compute"] += 1
            self._record_pim_compute_duration()
            # self._record_use_col()
        elif inst["type"] == 0b01:
            self.per_class_cnt["pim"]["pim_set"] += 1
        elif inst["type"] == 0b10:
            self.per_class_cnt["pim"]["pim_output"] += 1
        elif inst["type"] == 0b11:
            self.per_class_cnt["pim"]["pim_transfer"] += 1
        else:
            assert False, f"Unknown pim inst type: {inst['type']}"

    def _record_simd_class_inst(self, inst):
        self.per_class_cnt["simd"]["total"] += 1
        if inst["opcode"] == 0:
            self.per_class_cnt["simd"]["add"] += 1
        elif inst["opcode"] == 1:
            self.per_class_cnt["simd"]["add-scalar"] += 1
        elif inst["opcode"] == 2:
            self.per_class_cnt["simd"]["multiply"] += 1
        elif inst["opcode"] == 3:
            self.per_class_cnt["simd"]["quantify"] += 1
        elif inst["opcode"] == 4:
            self.per_class_cnt["simd"]["quantify-resadd"] += 1
        elif inst["opcode"] == 5:
            self.per_class_cnt["simd"]["quantify-multiply"] += 1
        elif inst["opcode"] == 6:
            self.per_class_cnt["simd"]["max"] += 1
        elif inst["opcode"] == 7:
            self.per_class_cnt["simd"]["mul-scalar"] += 1
        elif inst["opcode"] == 8:
            self.per_class_cnt["simd"]["floor"] += 1
        else:
            assert False, f"Unknown simd opcode: {inst['opcode']}"

    def _record_scalar_class_inst(self, inst):
        self.per_class_cnt["scalar"]["total"] += 1
        if inst["type"] == 0b00:
            self.per_class_cnt["scalar"]["rr"] += 1
        elif inst["type"] == 0b01:
            self.per_class_cnt["scalar"]["ri"] += 1
        elif inst["type"] == 0b10:
            if inst["opcode"] == 0b00:
                self.per_class_cnt["scalar"]["load"] += 1
            elif inst["opcode"] == 0b01:
                self.per_class_cnt["scalar"]["store"] += 1
            else:
                assert False, f"Unknown scalar opcode: {inst['opcode']}"
        elif inst["type"] == 0b11:
            if inst["opcode"] == 0b00:
                self.per_class_cnt["scalar"]["general_li"] += 1
            elif inst["opcode"] == 0b01:
                self.per_class_cnt["scalar"]["special_li"] += 1
            elif inst["opcode"] in [0b10, 0b11]:
                self.per_class_cnt["scalar"]["special_general_assign"] += 1
            else:
                assert False, f"Unknown scalar opcode: {inst['opcode']}"
        else:
            assert False, f"Unknown scalar opcode: {inst['opcode']}"

    def _record_trans_class_inst(self, inst):
        self.per_class_cnt["trans"]["total"] += 1

    def _record_control_class_inst(self, inst):
        self.per_class_cnt["ctr"]["total"] += 1
        if inst["type"] in [0, 1, 2, 3]:
            self.per_class_cnt["ctr"]["branch"] += 1
        elif inst["type"] == 4:
            self.per_class_cnt["ctr"]["jump"] += 1
        else:
            assert False, f"Unknown control type: {inst['type']}"

    def _record_other_class_inst(self, inst):
        pass
        # self.per_class_cnt["other"]["total"] += 1

    def dump(self, save_path, prefix=""):
        pim_compute_duration_mean = (
            (sum(self._pim_compute_duration) / len(self._pim_compute_duration))
            if len(self._pim_compute_duration) > 0
            else 0
        )
        save_data = {
            "total": self.total_inst_cnt,
            "per_class_cnt": self.per_class_cnt,
            "trans_addr": {
                "addr_max": str(self._trans_addr_max),
                "addr_plus_size_max": str(self._trans_addr_plus_size_max),
            },
            "pim_compute_duration_mean": pim_compute_duration_mean,
            "pim_compute_duration": self._pim_compute_duration,
        }
        save_json_path = os.path.join(save_path, f"{prefix}stats_for_optimize.json")
        with open(save_json_path, "w") as f:
            json.dump(save_data, f, indent=2)
        # print(f"Stats for optimize saved to {save_json_path}")

        save_data = {"total": self.total_inst_cnt, **self.per_class_cnt}
        save_json_path = os.path.join(save_path, f"{prefix}stats.json")
        with open(save_json_path, "w") as f:
            json.dump(save_data, f, indent=2)
        # print(f"Stats saved to {save_json_path}")

        # save_json_path = os.path.join(save_path, f"{prefix}regs.json")
        # with open(save_json_path, "w") as f:
        #     for reg_data in self._reg_data:
        #         f.write(f"pc: {reg_data['pc']}, ins id: {reg_data['idx']}, general reg: {reg_data['general_rf']}\n")
        # print(f"Regs info saved to {save_json_path}")

        pimset_data = {
            # "col_use": self._col_use.item(),
            # "col_total": self._col_total.item(),
            # "col_use_rate": self._col_use.item() / self._col_total.item(),
            "pimset_mask": [str(mask) for mask in self._pimset_mask]
        }
        save_json_path = os.path.join(save_path, f"{prefix}pimset.json")
        with open(save_json_path, "w") as f:
            json.dump(pimset_data, f, indent=2)
        # print(f"PIMSet data saved to {save_json_path}")

        assert (
            len(self._macro_comp_use)
            == len(self._macro_col_use)
            == len(self._macro_cell_total)
        ), f"{self._macro_comp_use=}, {self._macro_col_use=}, {self._macro_cell_total=}"
        macro_cell_use = [
            comp * col for comp, col in zip(self._macro_comp_use, self._macro_col_use)
        ]
        sum_macro_cell_use = sum(macro_cell_use)
        sum_macro_cell_total = sum(self._macro_cell_total)
        sample_macro_comp_use = (
            self._macro_comp_use
        )  # [:min(100,len(self._macro_comp_use))]
        sample_macro_col_use = (
            self._macro_col_use
        )  # [:min(100,len(self._macro_col_use))]
        cell_ultilization_data = {
            "use_rate": (
                sum_macro_cell_use / sum_macro_cell_total
                if sum_macro_cell_total > 0
                else 0
            ),
            "sum_macro_cell_use": sum_macro_cell_use,
            "sum_macro_cell_total": sum_macro_cell_total,
            # "macro_comp_col_use": [f"{comp}, {col}" for comp, col in zip(sample_macro_comp_use, sample_macro_col_use)],
        }
        save_json_path = os.path.join(save_path, f"{prefix}macro_ultilization.json")
        with open(save_json_path, "w") as f:
            json.dump(cell_ultilization_data, f, indent=2)
        # print(f"Cell ultilization data saved to {save_json_path}")

        logger.info(f"Stats files save to {save_path}")
