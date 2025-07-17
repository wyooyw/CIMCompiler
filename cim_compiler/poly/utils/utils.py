from collections import OrderedDict
from functools import partial

import islpy as isl
import yaml


def gen_ast(domain, schedule, context=None):
    if context is None:
        context = isl.Set("{ : }")
    build = isl.AstBuild.from_context(context)
    schedule = schedule.intersect_domain(domain)
    return build.node_from_schedule_map(schedule)


def gen_code(domain, schedule, context=None):
    return gen_ast(domain, schedule, context).to_C_str()


def print_code(domain, schedule, context=None):
    print(gen_code(domain, schedule, context))


def simplify_basic_map(isl_basic_map_, multi_value=False):
    assert isinstance(isl_basic_map_, isl.BasicMap)
    isl_map = isl_basic_map_.coalesce().remove_redundancies()

    if not multi_value:
        isl_map = isl_map.as_pw_multi_aff().as_map()

    assert len(isl_map.get_basic_maps()) == 1
    isl_basic_map = isl_map.get_basic_maps()[0]

    assert isinstance(isl_basic_map, isl.BasicMap), f"{type(isl_basic_map)}"
    return isl_basic_map


def rename_dims(isl_obj, dim_type, prefix):
    # assert isinstance(isl_obj, (isl.Map, isl.Set))
    for i in range(isl_obj.dim(dim_type)):
        isl_obj = isl_obj.set_dim_name(dim_type, i, "%s%d" % (prefix, i))
    return isl_obj


def rename_all_dims_for_basic_map(isl_basic_map):
    assert isinstance(isl_basic_map, isl.BasicMap)
    isl_basic_map = rename_dims(isl_basic_map, isl.dim_type.in_, prefix="i")
    isl_basic_map = rename_dims(isl_basic_map, isl.dim_type.out, prefix="o")
    return isl_basic_map


def rename_all_dims_for_basic_set(isl_basic_set, prefix="i"):
    assert type(isl_basic_set) in [isl.BasicSet, isl.Set], f"{type(isl_basic_set)=}"
    isl_basic_set = rename_dims(isl_basic_set, isl.dim_type.set, prefix=prefix)
    return isl_basic_set


def has_dim_name_for_basic_set(isl_basic_set):
    assert isinstance(isl_basic_set, isl.BasicSet)
    for i in range(isl_basic_set.n_dim()):
        if not isl_basic_set.has_dim_name(isl.dim_type.set, i):
            return False
    return True


def get_bound_for_all_dims(isl_basic_set):
    assert has_dim_name_for_basic_set(isl_basic_set)
    # print(isl_basic_set)
    # ipdb.set_trace()
    all_bounds = dict()
    for i in range(isl_basic_set.n_dim()):
        axis_name = isl_basic_set.get_dim_name(isl.dim_type.set, i)
        bounds = [None, None]  # [lower_bound, upper_bound]
        project_result = isl_basic_set.project_out_except(
            names=[axis_name], types=[isl.dim_type.set]
        )
        constrains = project_result.get_constraints()
        if len(constrains) == 1:
            assert constrains[0].is_equality()
            constraint = constrains[0]
            const_val = int(constraint.get_constant_val().to_str())
            bounds[0] = const_val
            bounds[1] = const_val
        elif len(constrains) == 2:
            for constraint in project_result.get_constraints():
                coef = constraint.get_coefficients_by_name()
                assert axis_name in coef
                coef_k = int(coef[axis_name].to_str())
                coef_b = int(constraint.get_constant_val().to_str())
                bound = -coef_b / coef_k
                bounds[coef_k < 0] = bound
        else:
            assert False, constrains
        all_bounds[axis_name] = bounds
    return all_bounds


def extract_index_set(acc_rel_map):
    acc_rel_map = rename_all_dims_for_basic_map(acc_rel_map)
    index_set = set()
    for constraint in acc_rel_map.get_constraints():
        if constraint.is_equality():
            coefficients = constraint.get_coefficients_by_name()
            for key, value in coefficients.items():
                if type(key) == str and key[0] == "i":
                    index_set.add(key)
    return index_set


def convert_index_set_to_bitmap(output_index_set, weight_index_set, feature_index_set):
    bitmap = dict()
    for index in feature_index_set.union(weight_index_set).union(output_index_set):
        bitmap[index] = [0, 0, 0]
    for index in output_index_set:
        bitmap[index][0] = 1
    for index in weight_index_set:
        bitmap[index][1] = 1
    for index in feature_index_set:
        bitmap[index][2] = 1
    return bitmap


def get_box_hull_shape(buffer):
    assert buffer.is_bounded(), f"{buffer=}"
    shape = [
        buffer.dim_max_val(i) - buffer.dim_min_val(i) + 1
        for i in range(buffer.dim(isl.dim_type.set))
    ]
    shape = [val_to_int(val) for val in shape]
    return shape


def rename_out_dims_for_basic_map(isl_basic_map, out_prefix="o"):
    assert isinstance(isl_basic_map, isl.BasicMap)
    isl_basic_map = rename_dims(isl_basic_map, isl.dim_type.out, prefix=out_prefix)
    return isl_basic_map


def val_to_int(val):
    return int(str(val))


unique_name_idx = 0
char_set = "abcdefghijklmnopqrstuvwxyz"


def get_unique_name():
    global unique_name_idx

    idx = unique_name_idx
    char_set_len = len(char_set)
    name = "_"
    while True:
        char = idx % char_set_len
        idx = idx // char_set_len
        name += char_set[char]
        if idx == 0:
            break

    unique_name_idx += 1
    return name


def make_pw_affs_to_aff_list(pw_affs):
    pw_aff_list = isl.PwAffList.alloc(pw_affs[0].get_ctx(), 0)
    for pw_aff in pw_affs:
        pw_aff_list = pw_aff_list.add(pw_aff)
    return pw_aff_list


def multi_pw_aff_from_pw_affs(pw_affs):
    pw_aff_list = make_pw_affs_to_aff_list(pw_affs)
    multi_aff_space = pw_affs[0].space.insert_dims(
        isl.dim_type.out, 0, len(pw_affs) - 1
    )
    multi_pw_aff = isl.MultiPwAff.from_pw_aff_list(multi_aff_space, pw_aff_list)
    return multi_pw_aff


def mpf_upper_bound_for_basic_set(multi_pw_aff, basic_set, level):
    """
    basic_set: [i0, i1, ..., i_{n-level-1}, i_{n-level}, ..., i_{n-1}]
    set upper bound for [i_{n-level}, ..., i_{n-1}]
    """
    n_dim = basic_set.dim(isl.dim_type.set)

    basic_map = isl.Map.from_range(basic_set)
    basic_map = basic_map.move_dims(
        isl.dim_type.in_, 0, isl.dim_type.out, 0, n_dim - level
    )
    basic_map = basic_map.upper_bound_multi_pw_aff(multi_pw_aff)
    basic_map = basic_map.move_dims(
        isl.dim_type.out, 0, isl.dim_type.in_, 0, n_dim - level
    )
    basic_set = basic_map.range()

    return basic_set


def zero_lower_bound_for_basic_set(basic_set, level):
    n_dim = basic_set.dim(isl.dim_type.set)
    for i in range(n_dim - level, n_dim):
        basic_set = basic_set.lower_bound_val(
            isl.dim_type.set, i, isl.Val.zero(basic_set.get_ctx())
        )
    return basic_set


def identity_map_from_set(domain):
    space = isl.Map.from_domain(domain).space
    n_dim_domain = domain.dim(isl.dim_type.set)
    space = space.insert_dims(isl.dim_type.out, 0, n_dim_domain)
    identity_schedule = isl.Map.identity(space)
    return identity_schedule


def get_range_dim_dynamic_size(acc_rel, pos):
    assert type(acc_rel) in (isl.BasicMap, isl.Map)
    dim_min_pw_aff = acc_rel.dim_min(pos)
    dim_max_pw_aff = acc_rel.dim_max(pos)

    dim_size = dim_max_pw_aff.sub(dim_min_pw_aff)
    dim_size = dim_size.add_constant_val(isl.Val.one(dim_size.get_ctx()))

    return dim_size


def get_mpf_lb_up_from_domain(domain, level):
    """
    basic_set: [i0, i1, ..., i_{n-level-1}, i_{n-level}, ..., i_{n-1}]
    set upper bound for [i_{n-level}, ..., i_{n-1}]
    """
    n_dim = domain.dim(isl.dim_type.set)

    basic_map = isl.Map.from_domain(domain)
    basic_map = basic_map.move_dims(isl.dim_type.out, 0, isl.dim_type.in_, level, 1)
    # if level==4:
    dim_min_pw_aff = basic_map.dim_min(0)
    dim_max_pw_aff = basic_map.dim_max(0)

    return dim_min_pw_aff, dim_max_pw_aff


def get_dynamic_shape_from_dynamic_map(isl_map):
    """
    range shape contains variable in domain
    """
    # num dim in out
    n_out = isl_map.dim(isl.dim_type.out)
    shape = [get_range_dim_dynamic_size(isl_map, pos) for pos in range(n_out)]
    return shape


def make_pw_affs_to_aff_list(pw_affs):
    pw_aff_list = isl.PwAffList.alloc(pw_affs[0].get_ctx(), 0)
    for pw_aff in pw_affs:
        pw_aff_list = pw_aff_list.add(pw_aff)
    return pw_aff_list


def get_static_box_shape(buffer):
    assert buffer.is_box(), f"{buffer=}"
    shape = [
        buffer.dim_max_val(i) - buffer.dim_min_val(i) + 1
        for i in range(buffer.dim(isl.dim_type.set))
    ]
    shape = [val_to_int(val) for val in shape]
    return shape


def ordered_yaml_load(s, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping
    )
    # with open(yaml_path) as stream:
    return yaml.load(s, OrderedLoader)


def ordered_yaml_dump(data, stream=None, Dumper=yaml.SafeDumper, **kwds):
    class OrderedDumper(Dumper):
        pass

    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, data.items()
        )

    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)


def pretty_print_schedule_tree(node):
    s = str(node)
    a = ordered_yaml_load(s)
    print(ordered_yaml_dump(a, indent=2))
    print("\n")


def print_schedule_tree_as_code(node, context=None):
    while node.get_tree_depth() > 0:
        node = node.parent()

    domain = node.domain_get_domain()
    schedule = node.get_subtree_schedule_union_map()
    print("\n[ Domain and Schedule ]\n")
    print("domain:", domain)
    print("schedule:", schedule)

    print("\n[ Code ]\n")
    print_code(domain, schedule, context)


def get_root(node):
    while node.get_tree_depth() > 0:
        node = node.parent()
    return node


def get_dominate_iters_of_pw_multi_aff(pw_multi_aff, return_name=True):
    """
    {[i0,i1,..,ik] -> [f(i1,i2)]}
    return {i1,i2}
    """
    dim_names = [
        pw_multi_aff.get_dim_name(isl.dim_type.in_, i)
        for i in range(pw_multi_aff.dim(isl.dim_type.in_))
    ]
    n_dim_range = pw_multi_aff.dim(isl.dim_type.out)

    dominate_dims = set()

    for cond, multi_aff in get_pieces_from_pw_multi_aff(pw_multi_aff):
        for dim in range(n_dim_range):
            aff = multi_aff.get_at(dim)
            for i in range(aff.dim(isl.dim_type.in_)):
                # coef = aff.get_coefficient_val(isl.dim_type.in_, i)
                if aff.involves_dims(isl.dim_type.in_, i, 1):
                    dominate_dims.add(dim_names[i] if return_name else i)

    return dominate_dims


def get_pieces_from_pw_multi_aff(pw_multi_aff):
    record = []
    pw_multi_aff.foreach_piece(lambda x, y: record.append((x, y)))
    return record


def get_dominate_iters_of_pw_multi_aff_per_out(pw_multi_aff, return_name=True):
    """
    {[i0,i1,..,ik] -> [f(i1,i2)]}
    return {i1,i2}
    """
    dim_names = [
        pw_multi_aff.get_dim_name(isl.dim_type.in_, i)
        for i in range(pw_multi_aff.dim(isl.dim_type.in_))
    ]
    n_dim_range = pw_multi_aff.dim(isl.dim_type.out)

    dominate_dims = []
    for i in range(n_dim_range):
        dominate_dims.append(set())

    for cond, multi_aff in get_pieces_from_pw_multi_aff(pw_multi_aff):
        for dim in range(n_dim_range):
            aff = multi_aff.get_at(dim)
            for i in range(aff.dim(isl.dim_type.in_)):
                if aff.involves_dims(isl.dim_type.in_, i, 1):
                    dominate_dims[dim].add(dim_names[i] if return_name else i)

    return dominate_dims


def record_points(point, record):
    multi_val = point.get_multi_val()
    val = [int(str(multi_val.get_val(i))) for i in range(len(multi_val))]
    record.append(val)


def get_points(set_):
    points = []
    record_points_fn = partial(record_points, record=points)
    set_.foreach_point(record_points_fn)
    return points


def show_set(set_):
    points = get_points(set_)
    for i, point in enumerate(points):
        print(f"{i}: {point}")


def show_map(map_):
    n_domain = map_.domain().dim(isl.dim_type.set)
    points = get_points(map_.wrap())
    for i, point in enumerate(points):
        lhs = point[:n_domain]
        rhs = point[n_domain:]
        print(f"{i}: {lhs} -> {rhs}")
