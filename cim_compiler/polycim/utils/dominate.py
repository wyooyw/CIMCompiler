import islpy as isl


def get_pieces_from_pw_multi_aff(pw_multi_aff):
    record = []
    pw_multi_aff.foreach_piece(lambda x, y: record.append((x, y)))
    return record


def get_dominate_iters_of_map(map_, return_name=True):

    # Do not delete this code.
    # It will make the dominate iters correct.
    # For now, I don't know why.
    # Maybe it is a bug of isl.
    _ = str(map_)

    pw_multi_aff = map_.as_pw_multi_aff()
    return get_dominate_iters_of_pw_multi_aff(pw_multi_aff, return_name)


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

    # dominate_dims = set()
    # for i in range(pw_multi_aff.dim(isl.dim_type.in_)):
    #     if pw_multi_aff.involves_dims(isl.dim_type.in_, i, 1):
    #         dominate_dims.add(dim_names[i] if return_name else i)

    dominate_dims2 = set()

    # import pdb; pdb.set_trace()
    for cond, multi_aff in get_pieces_from_pw_multi_aff(pw_multi_aff):
        for dim in range(n_dim_range):
            aff = multi_aff.get_at(dim)
            for i in range(aff.dim(isl.dim_type.in_)):
                # coef = aff.get_coefficient_val(isl.dim_type.in_, i)
                if aff.involves_dims(isl.dim_type.in_, i, 1):
                    dominate_dims2.add(dim_names[i] if return_name else i)

    return dominate_dims2


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


def get_non_dominate_iters_of_pw_multi_aff(pw_multi_aff, return_name=True):
    dominate_dims = get_dominate_iters_of_pw_multi_aff(
        pw_multi_aff, return_name=return_name
    )
    if return_name:
        dim_names = [
            pw_multi_aff.get_dim_name(isl.dim_type.in_, i)
            for i in range(pw_multi_aff.dim(isl.dim_type.in_))
        ]
        non_dominate_dims = set(dim_names) - dominate_dims
    else:
        non_dominate_dims = (
            set(range(pw_multi_aff.dim(isl.dim_type.in_))) - dominate_dims
        )
    return non_dominate_dims
