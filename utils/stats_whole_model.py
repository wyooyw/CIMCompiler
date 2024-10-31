import json
import os
import sys


def sum_stats(stats_json_list):
    total_stats = dict()
    for stats_json in stats_json_list:
        with open(stats_json, "r") as f:
            stats = json.load(f)
        for key, value in stats.items():
            if type(value) == int:
                total_stats[key] = total_stats.get(key, 0) + value
            elif type(value) == dict:
                for sub_key, sub_value in value.items():
                    total_stats[key] = total_stats.get(key, dict())
                    total_stats[key][sub_key] = (
                        total_stats[key].get(sub_key, 0) + sub_value
                    )
    return total_stats


def stats_model(folder_path):
    # get all stats.json's path in given folder
    stats_json_list = []
    stats_without_dw_json_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == "stats.json":
                if "dwconv" not in root:
                    stats_without_dw_json_list.append(os.path.join(root, file))
                stats_json_list.append(os.path.join(root, file))

    total_stats = sum_stats(stats_json_list)
    total_stats_without_dw = sum_stats(stats_without_dw_json_list)

    # save total stats
    save_json_path = os.path.join(folder_path, "model_stats.json")
    with open(save_json_path, "w") as f:
        json.dump(total_stats, f, indent=2)

    save_json_path = os.path.join(folder_path, "model_stats_without_dw.json")
    with open(save_json_path, "w") as f:
        json.dump(total_stats_without_dw, f, indent=2)
    return total_stats, total_stats_without_dw


def stats_model_all_mode(model_path):
    stats_dense, stats_dense_without_dw = stats_model(os.path.join(model_path, "dense"))
    ratio = dict()
    ratio_without_dw = dict()
    for mode in ["value_sparse", "bit_sparse", "bit_value_sparse"]:
        mode_path = os.path.join(model_path, mode)
        if not os.path.exists(mode_path):
            continue
        stats, stats_without_dw = stats_model(os.path.join(model_path, mode))
        # print(f"{mode=}")
        # print(f"{stats=}")
        ratio[mode] = {
            "total": stats["total"] / stats_dense["total"],
            "pim": stats["pim"]["pim_compute"] / stats_dense["pim"]["pim_compute"],
        }
        ratio_without_dw[mode] = {
            "total": stats_without_dw["total"] / stats_dense_without_dw["total"],
            "pim": stats_without_dw["pim"]["pim_compute"]
            / stats_dense_without_dw["pim"]["pim_compute"],
        }
    save_json_path = os.path.join(model_path, "model_stats_ratio.json")
    with open(save_json_path, "w") as f:
        json.dump(ratio, f, indent=2)

    save_json_path = os.path.join(model_path, "model_stats_ratio_without_dw.json")
    with open(save_json_path, "w") as f:
        json.dump(ratio_without_dw, f, indent=2)


def sum_stats_for_macro_utilization(stats_json_list):
    total_stats = dict()
    for stats_json in stats_json_list:
        with open(stats_json, "r") as f:
            stats = json.load(f)
        assert stats["use_rate"] > 0, f"{stats_json} use_rate={stats['use_rate']}"
        assert (
            stats["sum_macro_cell_use"] > 0
        ), f"{stats_json} sum_macro_cell_use={stats['sum_macro_cell_use']}"
        assert (
            stats["sum_macro_cell_total"] > 0
        ), f"{stats_json} sum_macro_cell_total={stats['sum_macro_cell_total']}"
        for key, value in stats.items():
            total_stats[key] = total_stats.get(key, 0) + value

    return total_stats


def stats_macro_utilization(folder_path):
    # get all stats.json's path in given folder
    stats_json_list = []
    stats_json_list_no_dw = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == "macro_ultilization.json":
                if "dwconv" not in root:
                    stats_json_list_no_dw.append(os.path.join(root, file))
                stats_json_list.append(os.path.join(root, file))

    total_stats = sum_stats_for_macro_utilization(stats_json_list)
    total_stats["use_rate"] = (
        total_stats["sum_macro_cell_use"] / total_stats["sum_macro_cell_total"]
        if total_stats["sum_macro_cell_total"] > 0
        else 0
    )

    total_stats_no_dw = sum_stats_for_macro_utilization(stats_json_list_no_dw)
    total_stats_no_dw["use_rate"] = (
        total_stats_no_dw["sum_macro_cell_use"]
        / total_stats_no_dw["sum_macro_cell_total"]
        if total_stats_no_dw["sum_macro_cell_total"] > 0
        else 0
    )

    # save total stats
    save_json_path = os.path.join(folder_path, "model_macro_utilization.json")
    with open(save_json_path, "w") as f:
        json.dump(total_stats, f, indent=2)

    save_json_path = os.path.join(folder_path, "model_macro_utilization_no_dw.json")
    with open(save_json_path, "w") as f:
        json.dump(total_stats_no_dw, f, indent=2)
    return total_stats


def max_inst(stats_json_list):
    max_inst_cnt = 0
    for stats_json in stats_json_list:
        with open(stats_json, "r") as f:
            stats = json.load(f)
        if stats["total"] > max_inst_cnt:
            max_inst_cnt = stats["total"]
            layer_path = stats_json
    return max_inst_cnt, layer_path


def max_inst_per_layer(folder_path):
    # get all stats.json's path in given folder
    stats_json_list = []
    stats_without_dw_json_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file == "stats.json":
                if "dwconv" not in root:
                    stats_without_dw_json_list.append(os.path.join(root, file))
                stats_json_list.append(os.path.join(root, file))
    max_inst_cnt, layer_path = max_inst(stats_json_list)
    print(f"{max_inst_cnt}, {layer_path=}")


if __name__ == "__main__":
    stats_model(os.path.join(sys.argv[1], "bit_value_sparse"))
    # stats_model_all_mode(sys.argv[1])
    # stats_macro_utilization(os.path.join(sys.argv[1], "bit_value_sparse"))
    # stats_macro_utilization(os.path.join(sys.argv[1], "bit_sparse"))
    # max_inst_per_layer(sys.argv[1])
