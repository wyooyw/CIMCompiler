import cim_compiler.poly.op.benchmark as benchmark
import cim_compiler.poly.utils.utils as utils
from cim_compiler.poly.passes.hardware_mapping_pass import (
    get_coalescing_schedule_from_mapping,
    get_reverse_coalescing_schedule_from_mapping,
)


def test_coalescing():
    batch = 2
    oc = 8
    ic = 4
    oh = 8
    ow = 8
    kh = 3
    kw = 3
    op = benchmark.get_op_conv2d(
        b=batch, oc=oc, ic=ic, oh=oh, ow=ow, kh=kh, kw=kw, stride=1, virtual_axis=True
    )
    mapping = {"h0": ("s2", "s5", "s6"), "h1": ("s1", "s3", "s4")}
    coalescing_schedule = get_coalescing_schedule_from_mapping(mapping, op)
    new_op = op.apply_schedule(coalescing_schedule, skip_simplify=True)
    new_domain_shape = utils.get_box_hull_shape(new_op.domain)
    new_domain_shape = tuple(new_domain_shape)
    assert new_domain_shape == (
        batch,
        ic * kh * kw,
        oc * oh * ow,
    ), f"new_domain_shape: {new_domain_shape}, expected: {(batch, ic*kh*kw, oc*oh*ow)}"


def test_reverse_coalescing():
    op = benchmark.get_op_conv2d(
        b=2, oc=8, ic=4, oh=8, ow=8, kh=3, kw=3, stride=1, virtual_axis=True
    )
    mapping = {"h0": ("s2", "s5", "s6"), "h1": ("s1", "s3", "s4")}
    coalescing_schedule = get_coalescing_schedule_from_mapping(mapping, op)
    reverse_coalescing_schedule = get_reverse_coalescing_schedule_from_mapping(
        mapping, op
    )
    new_op = op.apply_schedule(
        coalescing_schedule,
        reverse_schedule=reverse_coalescing_schedule,
        skip_simplify=True,
    )
    new_op = new_op.apply_schedule(
        reverse_coalescing_schedule,
        reverse_schedule=coalescing_schedule,
        skip_simplify=True,
    )

    assert op.domain == new_op.domain

    access_I = op.access_I.intersect_domain(op.domain)
    new_access_I = new_op.access_I.intersect_domain(new_op.domain)
    assert access_I.is_equal(new_access_I)

    access_O = op.access_O.intersect_domain(op.domain)
    new_access_O = new_op.access_O.intersect_domain(new_op.domain)
    assert access_O.is_equal(new_access_O)

    access_W = op.access_W.intersect_domain(op.domain)
    new_access_W = new_op.access_W.intersect_domain(new_op.domain)
    assert access_W.is_equal(new_access_W)


if __name__ == "__main__":
    test_coalescing()
