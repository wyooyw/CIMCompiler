import os
import subprocess
import tempfile

import pytest


@pytest.mark.parametrize(
    "graph_instruction_path, config_path",
    [
        # resnet18
        (
            "graphs/instructions_resnet18_0.5x_load_time_T4_B8.json",
            "config/dac25/config_gs_4.json",
        ),
        (
            "graphs/instructions_resnet18_0.5x_load_time_T8_B8.json",
            "config/dac25/config_gs_8.json",
        ),
        (
            "graphs/instructions_resnet18_0.5x_load_time_T12_B8.json",
            "config/dac25/config_gs_12.json",
        ),
        (
            "graphs/instructions_resnet18_0.5x_load_time_T16_B8.json",
            "config/dac25/config_gs_16.json",
        ),
        # mobilenet
        (
            "graphs/instructions_mobilenet_0.5x_load_time_T4_B8.json",
            "config/dac25/config_gs_4.json",
        ),
        (
            "graphs/instructions_mobilenet_0.5x_load_time_T8_B8.json",
            "config/dac25/config_gs_8.json",
        ),
        (
            "graphs/instructions_mobilenet_0.5x_load_time_T12_B8.json",
            "config/dac25/config_gs_12.json",
        ),
        (
            "graphs/instructions_mobilenet_0.5x_load_time_T16_B8.json",
            "config/dac25/config_gs_16.json",
        ),
    ],
)
def test_cimflow_network(graph_instruction_path, config_path):
    POLYCIM_HOME = os.environ.get("POLYCIM_HOME")
    graph_instruction_path = os.path.join(POLYCIM_HOME, graph_instruction_path)
    config_path = os.path.join(POLYCIM_HOME, config_path)

    with tempfile.TemporaryDirectory() as temp_dir:
        cmd = [
            "polycim",
            "cimflow_network",
            "-i",
            graph_instruction_path,
            "-o",
            temp_dir,
            "-c",
            config_path,
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    test_cimflow_network(
        "graphs/instructions_resnet18_0.5x_load_time_T4_B8.json",
        "config/dac25/config_gs_4.json",
    )
    pass
