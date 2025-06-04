import islpy as isl

from polycim.utils.utils import *


class FrameInfo:
    def __init__(self, input, output, macro):
        self.input = input
        self.output = output
        self.macro = macro

    def print(self, brief=False):
        print(self.get_str(brief=brief))

    def get_str(self, brief=False):
        j = len(self.output)
        k = len(self.input)
        # Prepare data
        self.input_for_print = [None for _ in range(k)]
        for i in range(k):
            if self.input[i] is not None:
                self.input_for_print[i] = (
                    "*" if brief else f"I[{','.join(self.input[i])}]"
                )
            else:
                self.input_for_print[i] = "0"
        self.output_for_print = [None for _ in range(j)]
        for i in range(j):
            if self.output[i] is not None:
                self.output_for_print[i] = (
                    "*" if brief else f"O[{','.join(self.output[i])}]"
                )
            else:
                self.output_for_print[i] = "0"
        self.macro_for_print = [[None for _ in range(j)] for _ in range(k)]
        for i1 in range(k):
            for i2 in range(j):
                if self.macro[i1][i2] is not None:
                    self.macro_for_print[i1][i2] = (
                        "*" if brief else f"W[{','.join(self.macro[i1][i2])}]"
                    )
                else:
                    self.macro_for_print[i1][i2] = "0"

        gap = "{:2}" if brief else "{:12}"

        s = ""
        # Print
        s += "Input:\n"
        for i in range(k):
            s += gap.format(self.input_for_print[i])
        s += "\n"
        s += "\nMacro:\n"
        for i1 in range(k):
            for i2 in range(j):
                s += gap.format(self.macro_for_print[i1][i2])
            s += "\n"
        s += "\nOutput:\n"
        for i in range(j):
            s += gap.format(self.output_for_print[i])
        s += "\n"
        return s


class VideoInfo:
    def __init__(self, frame_info_list):
        self.frame_info_list = frame_info_list


def count_time_list(domain, n_time_dim):
    if type(domain) == isl.UnionSet:
        domain = domain.as_set()
    elif not type(domain) == isl.Set:
        assert False, "domain should be UnionSet or Set!"

    domain = rename_all_dims_for_basic_set(domain, "i")
    n_dim = domain.n_dim()
    domain = domain.project_out_except(
        names=[f"i{i}" for i in range(n_time_dim)], types=[isl.dim_type.set]
    )
    return domain.count_val()


def extract_time_list(domain, n_time_dim):
    if type(domain) == isl.UnionSet:
        domain = domain.as_set()
    elif not type(domain) == isl.Set:
        assert False, "domain should be UnionSet or Set!"

    domain = rename_all_dims_for_basic_set(domain, "i")
    n_dim = domain.n_dim()
    domain = domain.project_out_except(
        names=[f"i{i}" for i in range(n_time_dim)], types=[isl.dim_type.set]
    )
    time_list = []

    def record(point):
        multi_val = point.get_multi_val()
        timestamp = []
        for i in range(len(multi_val)):
            timestamp.append(str(multi_val.get_val(i)))
        for i in range(len(multi_val), n_dim):
            timestamp.append(f"i{i}")
        timestamp = isl.Set(f"{{ [{','.join(timestamp)}] }}")
        time_list.append(timestamp)

    domain.foreach_point(record)
    return time_list


def extract_val_from_singleton_set(singleton_set):
    result = []

    def record(point):
        multi_val = point.get_multi_val()
        val = [str(multi_val.get_val(i)) for i in range(len(multi_val))]
        result.append(val)

    singleton_set.foreach_point(record)
    if len(result) > 0:
        return result[0]
    else:
        return None


def _extract_frame_info(
    domain, acc_rel_input, acc_rel_macro, acc_rel_output, timestamp, macro_j, macro_k
):
    """
    ret: frame(input, output, macro) at this timestamp. What data access in this timestamp.

    We assume that schedule is an identity schedule.
    """
    acc_rel_input = acc_rel_input.as_map()
    acc_rel_macro = acc_rel_macro.as_map()
    acc_rel_output = acc_rel_output.as_map()
    domain_n_dim = domain.n_dim()
    hardware_n_dim = 2
    # print(type(domain),domain)
    # print(type(timestamp),timestamp)
    # ipdb.set_trace()
    domain = domain.intersect(timestamp)
    input_data = [None for _ in range(macro_k)]
    macro_data = [[None for _ in range(macro_j)] for _ in range(macro_k)]
    output_data = [None for _ in range(macro_j)]

    # import pdb; pdb.set_trace()
    def record(point):
        multi_val = point.get_multi_val()
        domain_frame = isl.Set(
            f"{{ [{','.join([str(multi_val.get_val(i)) for i in range(domain_n_dim)])}] }}"
        )
        # print("domain_frame:",domain_frame)
        # print("acc_rel_input:",acc_rel_input, acc_rel_input.is_single_valued())
        input_point_data = (
            acc_rel_input.intersect_domain(domain_frame).range().remove_redundancies()
        )
        macro_point_data = (
            acc_rel_macro.intersect_domain(domain_frame).range().remove_redundancies()
        )
        output_point_data = (
            acc_rel_output.intersect_domain(domain_frame).range().remove_redundancies()
        )
        # print("input_point_data:",input_point_data, input_point_data.is_singleton())
        assert (
            input_point_data.is_singleton()
        ), f"input_data should be singleton, but {input_point_data}!"
        assert (
            macro_point_data.is_singleton()
        ), f"macro_data should be singleton, but {macro_point_data}!"
        assert (
            output_point_data.is_singleton()
        ), f"input_data should be singleton, but {output_point_data}!"
        pos_j = int(str(multi_val.get_val(domain_n_dim - 1)))
        pos_k = int(str(multi_val.get_val(domain_n_dim - 2)))
        # print("j:",pos_j," k:", pos_k)
        input_data[pos_k] = (
            extract_val_from_singleton_set(input_point_data)
            if input_data[pos_k] is None
            else input_data[pos_k]
        )
        macro_data[pos_k][pos_j] = extract_val_from_singleton_set(macro_point_data)
        output_data[pos_j] = (
            extract_val_from_singleton_set(output_point_data)
            if output_data[pos_j] is None
            else output_data[pos_j]
        )

    domain.foreach_point(record)
    # return None
    return FrameInfo(input_data, output_data, macro_data)


def schedule_identity(software_op):
    if not software_op.schedule.is_identity():
        # make schedule identity
        domain = software_op.domain
        schedule = software_op.schedule
        access_relations = software_op.access_relations

        new_domain = schedule.intersect_domain(domain).range()
        new_schedule = new_domain.identity()
        new_output_acc_rel = schedule.reverse().apply_range(
            access_relations.output_acc_rel
        )
        new_A_acc_rel = schedule.reverse().apply_range(access_relations.A_acc_rel)
        new_B_acc_rel = schedule.reverse().apply_range(access_relations.B_acc_rel)
        new_software_op = Operator(
            new_domain, new_output_acc_rel, new_A_acc_rel, new_B_acc_rel, new_schedule
        )
        # ipdb.set_trace()
        software_op = new_software_op
    return software_op


def get_macro_hash(macro):
    """
    macro is a 2d list, such as:
    [[None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None], [None, None, None, None], [['0', '0'], None, None, None], [['1', '0'], ['0', '0'], None, None], [['2', '0'], ['1', '0'], None, None], [['3', '0'], ['2', '0'], None, None]]
    Each element is None or a list of string.
    make macro to a hash string.
    """
    return hash(str(macro))


def extract_frame_info(software_op, cim_cfg, different_weight=False):
    #
    # if not software_op.schedule.is_identity():
    # software_op = schedule_identity(software_op)

    # get time stamp
    domain_set = software_op.domain.as_set()
    time_list = extract_time_list(domain_set, domain_set.n_dim() - 2)

    macro_hash_list = set()

    for timestamp in time_list:

        frame_info = _extract_frame_info(
            domain=domain_set,
            acc_rel_input=software_op.access_I,
            acc_rel_macro=software_op.access_W,
            acc_rel_output=software_op.access_O,
            timestamp=timestamp,
            macro_j=cim_cfg.n_group_vcol,
            macro_k=cim_cfg.n_comp,
        )
        macro_hash = get_macro_hash(frame_info.macro)
        if different_weight and macro_hash in macro_hash_list:
            continue
        else:
            macro_hash_list.add(macro_hash)
            yield timestamp, frame_info


def draw(min_compute_op, cim_cfg, brief=False, different_weight=True):
    for idx, value in enumerate(
        extract_frame_info(min_compute_op, cim_cfg, different_weight=different_weight)
    ):
        timestamp, frame_info = value
        print(f"Index: {idx}.    Timestamp: {timestamp}")
        frame_info.print(brief=brief)
        c = input("continue?(y/n):")
        if c == "n":
            break
        else:
            continue


if __name__ == "__main__":
    domain = isl.Set(
        " { feature[ic = 0, ih, iw = 0] : (ih) mod 2 = 0 and 0 <= ih <= 1 }"
    )

    print(domain.is_singleton())
    exit()
    # schedule = isl.BasicMap("{ S[i,j] -> [i,j] }")
    # print(domain.is_singleton())
    # exit()
    # schedule = isl.UnionMap("{ S[i,j] -> [i+j,j]}")
    # extract_frame_info(domain, schedule)
    time_list = extract_time_list(domain, 2)
    print(time_list)
    extract_frame_info(domain, None, None, None, time_list[0])
    # print(schedule.intersect_domain(domain))
