import os
from functools import partial

import islpy as isl
import matplotlib.pyplot as plt
from islplot.plotter import plot_map_as_groups, plot_set_points

from polycim.op.base_operator import BasicOperator
from polycim.passes.affine_transform_pass import auto_skewing_pass


def count(basic_set):
    cnt = 0

    def record(point):
        nonlocal cnt
        cnt += 1
        print("\t", point.get_multi_val().get_val(0))

    basic_set.foreach_point(record)
    return cnt


def conv1d(schedule=False):
    domain = isl.BasicSet("{ S[i,k]: 0<=i<8 and 0<=k<3 }")
    acc_rel_I = isl.BasicMap("{ S[i,k] -> A[i+k] }")

    if schedule:
        schedule = isl.BasicMap("{ S[i,k] -> S[i,i+k] }")

        concrete_schedule = schedule.intersect_domain(domain)
        domain = concrete_schedule.range()

        acc_rel_I = concrete_schedule.reverse().apply_range(acc_rel_I)
    return domain, acc_rel_I


def simplify_basic_map(isl_basic_map, multi_value=False):
    assert isinstance(isl_basic_map, isl.BasicMap)
    isl_basic_map = isl_basic_map.coalesce().remove_redundancies()

    if not multi_value:
        isl_basic_map = isl_basic_map.as_pw_multi_aff().as_map()

    assert len(isl_basic_map.get_basic_maps()) == 1
    isl_basic_map = isl_basic_map.get_basic_maps()[0]

    assert isinstance(isl_basic_map, isl.BasicMap), f"{type(isl_basic_map)}"
    return isl_basic_map


def conv1d_tile(schedule=False):
    operator = BasicOperator(
        domain=isl.BasicSet("{ [i,k]: 0<=i<4 and 0<=k<3 }"),
        access_I=isl.BasicMap("{ [i,k] -> I[i+k] }"),
        access_O=isl.BasicMap("{ [i,k] -> O[i] }"),
        access_W=isl.BasicMap("{ [i,k] -> W[k] }"),
    )

    if schedule:
        schedule_1 = isl.BasicMap("{ [i,k] -> [floor(i/4),(i%4), k] }")
        schedule_2 = isl.BasicMap("{ [io,ii,k] -> [io,ii,ii+k] }")
        schedule = schedule_1  # .apply_range(schedule_2)  # .apply_range(schedule_3)

        operator = operator.apply_schedule(schedule)

    return operator


def conv1d_stride2(schedule=False):
    domain = isl.BasicSet("{ S[i,k]: 0<=i<8 and 0<=k<6 }")
    acc_rel_I = isl.BasicMap("{ S[i,k] -> A[2*i+k] }")

    if schedule:
        schedule_1 = isl.BasicMap("{ S[i,k] -> S[(k%2), i,floor(k/2)] }")
        schedule_2 = isl.BasicMap("{ S[ko,i,ki] -> S[ko,i,i+ki] }")
        schedule = schedule_1.apply_range(schedule_2)

        # transform by scheudle
        concrete_schedule = schedule.intersect_domain(domain)
        domain = concrete_schedule.range()

        acc_rel_I = concrete_schedule.reverse().apply_range(acc_rel_I)
    return domain


def conv1d_dialeted2(schedule=False):
    operator = BasicOperator(
        domain=isl.BasicSet("{ [i,k]: 0<=i<8 and 0<=k<3 }"),
        access_I=isl.BasicMap("{ [i,k] -> I[i+2*k] }"),
        access_O=isl.BasicMap("{ [i,k] -> O[i] }"),
        access_W=isl.BasicMap("{ [i,k] -> W[k] }"),
    )

    if schedule:
        schedule_1 = isl.BasicMap("{ [i,k] -> [floor(i/2),(i%2), k] }")
        # schedule_2 = isl.BasicMap("{ S[ii,io,k] -> S[io,ii,ii+k] }")
        schedule = schedule_1  # .apply_range(schedule_2)

        operator = operator.apply_schedule(schedule)

    return operator


def conv1d_stride2_dialeted2(schedule=False):
    operator = BasicOperator(
        domain=isl.BasicSet("{ [i,k]: 0<=i<8 and 0<=k<6 }"),
        access_I=isl.BasicMap("{ [i,k] -> I[2*i+2*k] }"),
        access_O=isl.BasicMap("{ [i,k] -> O[i] }"),
        access_W=isl.BasicMap("{ [i,k] -> W[k] }"),
    )

    if schedule:
        schedule_1 = isl.BasicMap("{ [i,k] -> [(i%2), floor(i/2), k] }")
        # schedule_2 = isl.BasicMap("{ S[io,ii,k] -> S[io,ii,ii+k] }")
        schedule = schedule_1  # .apply_range(schedule_2)

        operator = operator.apply_schedule(schedule)

    return operator


def record_points(point, record, tuple_name):
    if tuple_name is None:
        tuple_name = ""
    multi_val = point.get_multi_val()
    val = [str(multi_val.get_val(i)) for i in range(len(multi_val))]
    point = isl.BasicSet("{ " + tuple_name + "[" + ",".join(val) + "] }")
    record.append(point)


def foreach_outer_iters(domain, level=2):
    n_dim = domain.dim(isl.dim_type.set)
    assert level <= n_dim, f"{level=} should be less or equal than {n_dim=}!"
    tuple_name = domain.get_tuple_name()
    outer_domain = domain.project_out(isl.dim_type.set, n_dim - level, level)
    points = []

    record_points_fn = partial(record_points, record=points, tuple_name=tuple_name)
    outer_domain.foreach_point(record_points_fn)
    for point in points:
        whole_point = point.insert_dims(isl.dim_type.set, n_dim - level, level)
        if tuple_name:
            whole_point = whole_point.set_tuple_name(tuple_name)
        subdomain = domain.intersect(whole_point)
        subdomain = subdomain.project_out(isl.dim_type.set, 0, n_dim - level)

        if tuple_name:
            subdomain = subdomain.set_tuple_name(tuple_name)
        yield point, whole_point, subdomain


def get_access_points(domain, access):
    points = []
    concrete_access = access.intersect_domain(domain)
    record_fn = partial(
        record_points, record=points, tuple_name=access.get_tuple_name(isl.dim_type.out)
    )
    concrete_access.range().foreach_point(record_fn)
    return points


def draw_access_points(
    access, access_points, padded_outer_point, save_dir, name, figsize=(6, 6)
):

    plt.title(name)
    plt.gca().set_aspect(1)
    plt.tight_layout()
    plt.xticks(list(range(32)))
    plt.yticks(list(range(32)))
    plt.grid(True)

    for idx, access_point in enumerate(access_points):
        domain_points = access.reverse().intersect_domain(access_point).range()
        n_dim = domain_points.dim(isl.dim_type.set)
        domain_points = domain_points.intersect(padded_outer_point)
        domain_points = domain_points.project_out(isl.dim_type.set, 0, n_dim - 2)
        plot_set_points(domain_points, color=color_list[idx], size=10)


def draw_operator(save_dir, operator):
    domain = operator.domain
    domain_sizes = [
        int(str(domain.dim_max_val(i))) + 1 for i in range(domain.dim(isl.dim_type.set))
    ]

    access_I = operator.concrete_access_I()
    access_O = operator.concrete_access_O()

    access_I_points = get_access_points(domain, access_I)
    access_O_points = get_access_points(domain, access_O)

    for outer_point, padded_outer_point, inner_domain in foreach_outer_iters(domain, 2):
        plt.figure(figsize=(domain_sizes[-2] * 2, domain_sizes[-1] * 3))

        plt.subplot(1, 2, 1)
        draw_access_points(
            access=access_I,
            access_points=access_I_points,
            padded_outer_point=padded_outer_point,
            save_dir=save_dir,
            name="Inputs",
        )

        plt.subplot(1, 2, 2)
        draw_access_points(
            access=access_O,
            access_points=access_O_points,
            padded_outer_point=padded_outer_point,
            save_dir=save_dir,
            name="Outputs",
        )

        plt.savefig(os.path.join(save_dir, str(outer_point)))
        plt.clf()


def remove_dir_contents(dir_path):
    for root, dirs, files in os.walk(
        dir_path, topdown=False
    ):  # topdown=False从目录树的底部开始遍历
        for name in files:
            os.remove(os.path.join(root, name))  # 删除文件
        for name in dirs:
            os.rmdir(os.path.join(root, name))  # 删除空目录


"""
1D conv
for i in range(8):
    for k in rang(3):
        C[i] += A[i+k] * B[k]
"""
color_list = [
    "#FF0000",
    "#00FF00",
    "#0000FF",
    "#FFFF00",
    "#FF00FF",
    "#00FFFF",
    "#FF8000",
    "#8000FF",
    "#FF0080",
    "#0080FF",
    "#00FF80",
    "#80FF00",
    "#FF80FF",
    "#80FFFF",
    "#FF8080",
    "#80FF80",
    "#8080FF",
    "#FF8080",
]

temp_dir = ".temp_imgs"
os.makedirs(temp_dir, exist_ok=True)
# clear files in temp_dir
remove_dir_contents(temp_dir)

operator = conv1d_tile(schedule=False)
# draw_operator(temp_dir, operator)
# operator = conv1d_tile(schedule=True)


schedule_1 = isl.BasicMap("{ [i,k] -> [floor(i/4),(i%4), k] }")
schedule_2 = isl.BasicMap("{ [io,ii,k] -> [io,ii,ii+k] }")
schedule_3 = isl.BasicMap("{ [a,b,c] -> [a,floor(b/4),floor(c/3)] }")
# schedule_1 = isl.BasicMap("{ [i,k] -> [i+k,i] }")
# schedule_2 = isl.BasicMap("{ [ik,k] -> [floor(ik/3),floor(k/4)] }")
schedule = schedule_1.apply_range(schedule_2).apply_range(schedule_3)
schedule = schedule.intersect_domain(operator.domain)

domain = operator.domain
domain_sizes = [
    int(str(domain.dim_max_val(i))) + 1 for i in range(domain.dim(isl.dim_type.set))
]
plt.figure(figsize=(domain_sizes[-2] * 2, domain_sizes[-1] * 3))
plt.gca().set_aspect(1)
plt.tight_layout()
plt.xticks(list(range(32)))
plt.yticks(list(range(32)))
plt.grid(True)
plot_map_as_groups(schedule, alpha=0.5, border=0.1, vertex_color="red")
plt.savefig("map.png")

exit()
skewed_operator_list, ori_op_list, schedule_list, base_matrix_list = auto_skewing_pass(
    op_list=[operator], max_reuse_factor_for_arrays=(16, 16), return_detail=True
)
for idx, new_op in enumerate(skewed_operator_list):
    save_dir = f".temp_imgs/{idx}"
    os.makedirs(save_dir, exist_ok=True)
    draw_operator(save_dir, new_op)

    ori_op = ori_op_list[idx]
    schedule = schedule_list[idx]
    base_matrix = base_matrix_list[idx]
    with open(f"{save_dir}/schedule.txt", "w") as f:
        f.write(f"{ori_op.domain = }\n")
        f.write(f"{ori_op.access_I = }\n")
        f.write(f"{ori_op.access_O = }\n")
        f.write("-----------------------------\n")
        f.write(f"{base_matrix=}\n")
        f.write(f"{schedule=}\n")
        f.write("-----------------------------\n")
        f.write(f"{new_op.domain = }\n")
        f.write(f"{new_op.access_I = }\n")
        f.write(f"{new_op.access_O = }\n")
