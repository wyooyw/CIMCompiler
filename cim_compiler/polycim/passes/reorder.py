import itertools

import islpy as isl

from polycim.utils.logger import level_tqdm


def reorder_outer(operator, inner_level):
    n_domain_iter = operator.domain.dim(isl.dim_type.set)
    n_outer_iter = n_domain_iter - inner_level

    domain_iter_names = operator.domain.get_var_names(isl.dim_type.set)

    outer_names = [domain_iter_names[i] for i in range(n_outer_iter)]
    inner_names = [domain_iter_names[n_outer_iter + i] for i in range(inner_level)]

    permutations = list(itertools.permutations(outer_names))

    new_operator_list = []
    for p in level_tqdm(permutations, desc="build_reorder"):
        reorder_schedule = isl.BasicMap(
            f"{{ [{','.join(domain_iter_names)}] -> [{','.join(list(p) + inner_names)}] }}"
        )
        new_operator = operator.apply_schedule(reorder_schedule, skip_simplify=True)
        yield new_operator
