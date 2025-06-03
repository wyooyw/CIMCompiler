import subprocess
from tqdm import tqdm

def run_single_exp(n_head, hidden_size, seqlen, mapping_cp_sizes, world_size, config_path, name_prefix):
    # python3 exp/llm_cim/main.py \
    # --n-head 4 \
    # --hidden-size 512 \
    # --seqlen 1025 \
    # --mapping-cp-sizes 1 \
    # --world-size 4 \
    # --config-path ${PWD}/test/op/llm/config.json \
    # --name-prefix opt_6b7_1024 \
    # --split-stages
    subprocess.run([
        "python3",
        "exp/llm_cim/main.py",
        "--n-head", str(n_head),
        "--hidden-size", str(hidden_size),
        "--seqlen", str(seqlen),
        "--mapping-cp-sizes", *[str(cp_size) for cp_size in mapping_cp_sizes],
        "--world-size", str(world_size),
        "--config-path", config_path,
        "--name-prefix", name_prefix,
        "--split-stages"
    ], check=True)

def run_exp_model(model_name, seqlen, mapping_cp_sizes):
    config_path = "/app/CIMCompiler/CIMCompiler/test/op/llm/config.json"
    world_size = 32
    if model_name == "opt_6b7":
        n_head = 32
        hidden_size = 4096
    elif model_name == "opt_13b":
        n_head = 40
        hidden_size = 5120
    elif model_name == "opt_30b":
        n_head = 56
        hidden_size = 7168
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    name_prefix = f"{model_name}_{seqlen}"

    run_single_exp(n_head, hidden_size, seqlen, mapping_cp_sizes, world_size, config_path, name_prefix)

def main():
    model_name = "opt_13b"
    seqlen_in = 64
    save_list = []
    for seqlen_out in tqdm([1, 16, 32, 64, 128, 256, 512, 1024]):
        seqlen = seqlen_in + seqlen_out - 1
        if seqlen <= 1024:
            mapping_cp_sizes = [1, 4]
        elif seqlen <= 2048:
            mapping_cp_sizes = [2, 2, 4]
        else:
            assert False, f"seqlen {seqlen} is too large"
        run_exp_model(model_name, seqlen, mapping_cp_sizes)
        save_list.append(f"result/{model_name}_{seqlen}")
    # create a tar file for all the result
    print("Save to:")
    for i, path in enumerate(save_list):
        print(f"{i}:  {path}")

def main2():
    seqlen_in = 512
    seqlen_out = 32
    seqlen = seqlen_in + seqlen_out - 1
    save_list = []

    model_name = "opt_6b7"
    mapping_cp_sizes = [1]
    run_exp_model(model_name, seqlen, mapping_cp_sizes)
    save_list.append(f"result/{model_name}_{seqlen}")

    model_name = "opt_13b"
    mapping_cp_sizes = [1, 4]
    run_exp_model(model_name, seqlen, mapping_cp_sizes)
    save_list.append(f"result/{model_name}_{seqlen}")

    model_name = "opt_30b"
    mapping_cp_sizes = [1, 2, 4]
    run_exp_model(model_name, seqlen, mapping_cp_sizes)
    save_list.append(f"result/{model_name}_{seqlen}")

    # create a tar file for all the result
    print("Save to:")
    for i, path in enumerate(save_list):
        print(f"{i}:  {path}")

if __name__ == "__main__":
    main2()