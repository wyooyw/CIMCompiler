import math
import os
import subprocess
import tempfile

import pandas as pd
import pytest


def ceil(a, b):
    return int(math.ceil(a / b))


@pytest.mark.parametrize(
    "cim_cfg_path, op_id, cim_count, axis_align",
    [
        *[
            ("c16b32.json", op_id, cim_count, axis_align)
            for op_id, cim_count, axis_align in [
                *[
                    ("C1", _cim_count, _axis_align)
                    for _cim_count, _axis_align in [(3136, False), (-1, True)]
                ],
                *[
                    ("C2", _cim_count, _axis_align)
                    for _cim_count, _axis_align in [(784, False), (-1, True)]
                ],
            ]
        ],
        *[
            ("g2m2c16b32.json", op_id, cim_count, False)
            for op_id, cim_count in [
                ("C1", 3136 // 2),
                ("C2", 784 // 2),
            ]
        ],
        *[
            ("g4m4c16b32.json", op_id, cim_count, False)
            for op_id, cim_count in [
                ("C1", 3136 // 4),
                ("C2", 784 // 4),
            ]
        ],
        *[
            ("c32b64.json", op_id, cim_count, False)
            for op_id, cim_count in [
                # small kernels
                ("C1", 1568),
                ("C2", 392),
                ("C4", 28),
                ("C7", 7),
                ("C12", 98),
                # large kernels
                ("C3", 196),
                ("C5", 56),
                ("C6", 14),
                ("C8", 1176),
                ("C9", 21),
                ("C10", 34496),
                ("C11", 56),
                ("C13", 224),
                # conv-3d
                ("C15", 21952),
            ]
        ],
        *[
            ("g4m4c32b64.json", op_id, cim_count, True)
            for op_id, cim_count in [
                # small kernels
                ("C1", -1),
                ("C2", -1),
                ("C4", -1),
                ("C7", -1),
                ("C12", -1),
                # large kernels
                ("C3", -1),
                ("C5", -1),
                ("C6", -1),
                ("C8", -1),
                ("C9", -1),
                ("C10", -1),
                ("C11", -1),
                ("C13", -1),
                # conv-3d
                ("C15", -1),
            ]
        ],
        *[
            ("g2m2c32b64.json", op_id, cim_count, False)
            for op_id, cim_count in [
                ("C1", 1568 // 2),
                ("C2", 392 // 2),
                ("C4", 28 // 2),
                ("C7", ceil(7, 2)),
                ("C12", 98 // 2),
            ]
        ],
        *[
            ("g4m4c32b64.json", op_id, cim_count, False)
            for op_id, cim_count in [
                ("C1", ceil(1568, 4)),
                ("C2", ceil(392, 4)),
                ("C4", ceil(28, 4)),
                ("C7", ceil(7, 4)),
                ("C12", -1),
            ]
        ],
        *[
            ("c64b64.json", op_id, cim_count, False)
            for op_id, cim_count in [
                # small kernels
                ("C1", 1568),
                ("C2", 392),
                ("C4", 28),
                ("C7", 7),
                ("C12", 98),
                # large kernels
                ("C3", 98),
                ("C5", 28),
                ("C6", 7),
                ("C8", 784),
                ("C9", 14),
                ("C10", 17248),
                ("C11", 28),
                ("C13", 112),
                # conv-3d
                ("C15", 10976),
            ]
        ],
    ],
)
def test_result(cim_cfg_path, op_id, cim_count, axis_align):
    # op_id, status = run_op(cim_cfg_path, op_id, cim_count, axis_align)
    # assert status == 3, f"{status=}"

    polycim_home = os.environ["POLYCIM_HOME"]
    compiler_cfg_path = os.path.join(
        polycim_home, "polycim/exp/iccad25/compiler_configs", cim_cfg_path
    )
    pimsim_cfg_path = os.path.join(
        polycim_home, "polycim/exp/iccad25/pimsim_configs", cim_cfg_path
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "polycim",
            "op",
            "--op-id",
            op_id,
            "--config-path",
            compiler_cfg_path,
            "--pimsim-cfg-path",
            pimsim_cfg_path,
            "--output-path",
            temp_dir,
            "--data-movement-full-vectorize",
            "--polycim",
            "--verify",
        ]
        if axis_align:
            cmd.append("--polycim-disable-affine")
        subprocess.run(cmd, check=True)

        # get result from result.csv
        result_path = os.path.join(temp_dir, "result.csv")
        df = pd.read_csv(result_path)
        cim_compute_ops = df.at[0, "cim_compute_ops"]
        if cim_count != -1:
            assert cim_compute_ops == cim_count, f"{cim_compute_ops=} != {cim_count=}"

        check_result = bool(df.at[0, "check_result"])
        assert check_result


if __name__ == "__main__":
    test_result("c32b64.json", "C2", -1, True)
