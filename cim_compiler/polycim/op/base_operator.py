import islpy as isl

import polycim.utils.utils as utils


class AccessRelation:
    def __init__(self, offsets, memory_name="global"):
        assert type(offsets) in (isl.BasicMap, isl.Map), f"{type(offsets)}"
        self.offsets = offsets
        self.memory_name = memory_name

    def convex_hull(self):
        return AccessRelation(self.offsets.convex_hull(), self.memory_name)

    def __repr__(self):
        return f"({self.memory_name}){self.offsets}"


class TensorAccessRelation(AccessRelation):
    def __init__(self, offsets, sizes, memory_name="global"):
        super().__init__(offsets, memory_name)
        self.sizes = sizes

    def convex_hull(self):
        return TensorAccessRelation(
            self.offsets.convex_hull(), self.sizes.convex_hull(), self.memory_name
        )


class Operator:
    def __init__(self):
        pass

    def apply_schedule(
        self, schedule, reverse_schedule=None, skip_simplify=False, name=None
    ):
        raise NotImplementedError


class BasicOperator(Operator):
    def __init__(
        self,
        domain,
        access_I,
        access_O,
        access_W,
        history_domains=None,
        history_schedules=None,
        attr=None,
    ):
        if history_domains is None:
            history_domains = list()
        if history_schedules is None:
            history_schedules = list()
        if attr is None:
            attr = dict()

        assert type(domain) == isl.BasicSet
        assert type(access_I) == isl.BasicMap
        assert type(access_O) == isl.BasicMap
        assert type(access_W) == isl.BasicMap
        self.domain = domain
        self.access_I = access_I
        self.access_O = access_O
        self.access_W = access_W
        self.history_domains = history_domains
        self.history_schedules = history_schedules

        self.rename_domain_and_access()

        self.attr = attr

    def rename_domain_and_access(self):
        self.domain = utils.rename_all_dims_for_basic_set(self.domain)
        self.access_I = utils.rename_all_dims_for_basic_map(self.access_I)
        self.access_O = utils.rename_all_dims_for_basic_map(self.access_O)
        self.access_W = utils.rename_all_dims_for_basic_map(self.access_W)

    def apply_schedule(
        self, schedule, reverse_schedule=None, skip_simplify=False, name=None
    ):
        assert type(schedule) == isl.BasicMap, f"{type(schedule)}"

        # transform by scheudle
        concrete_schedule = schedule.intersect_domain(self.domain)
        assert (
            concrete_schedule.reverse().is_single_valued()
        ), f"{concrete_schedule.reverse()}"
        domain = concrete_schedule.range()

        if reverse_schedule is None:
            reverse_schedule = concrete_schedule.reverse()
        else:
            assert reverse_schedule.dim(isl.dim_type.in_) == concrete_schedule.dim(
                isl.dim_type.out
            )
            assert reverse_schedule.dim(isl.dim_type.out) == concrete_schedule.dim(
                isl.dim_type.in_
            )
            reverse_schedule = reverse_schedule.intersect_domain(domain)

        access_I = reverse_schedule.apply_range(self.access_I)
        access_O = reverse_schedule.apply_range(self.access_O)
        access_W = reverse_schedule.apply_range(self.access_W)

        assert type(access_I) == isl.BasicMap, f"{type(access_I)}"
        assert type(access_O) == isl.BasicMap, f"{type(access_I)}"
        assert type(access_W) == isl.BasicMap, f"{type(access_I)}"

        if not access_O.is_single_valued():
            import pdb

            pdb.set_trace()

        if not skip_simplify:
            access_I = utils.simplify_basic_map(access_I)
            access_O = utils.simplify_basic_map(access_O)
            access_W = utils.simplify_basic_map(access_W)

        return BasicOperator(
            domain=domain,
            access_I=access_I,
            access_O=access_O,
            access_W=access_W,
            history_domains=[*self.history_domains, self.domain],
            history_schedules=[*self.history_schedules, {name: schedule}],
            attr={key: value for key, value in self.attr.items()},
        )

    def convex_hull(self):
        return BasicOperator(
            domain=self.domain.convex_hull(),
            access_I=self.access_I.convex_hull(),
            access_O=self.access_O.convex_hull(),
            access_W=self.access_W.convex_hull(),
            history_domains=[*self.history_domains, self.domain],
            history_schedules=[*self.history_schedules, "convex_hull"],
            attr={key: value for key, value in self.attr.items()},
        )

    def copy(self):
        return BasicOperator(
            domain=self.domain,
            access_I=self.access_I,
            access_O=self.access_O,
            access_W=self.access_W,
            history_domains=[*self.history_domains],
            history_schedules=[*self.history_schedules],
        )

    def get_access_by_name(self, buffer_name):
        if buffer_name == "I":
            return self.access_I
        elif buffer_name == "O":
            return self.access_O
        elif buffer_name == "W":
            return self.access_W
        else:
            raise Exception(f"Unknown buffer name: {buffer_name}")

    def concrete_access(self, access):
        concrete_access = access.intersect_domain(self.domain)
        return concrete_access

    def concrete_access_I(self):
        return self.concrete_access(self.access_I)

    def concrete_access_O(self):
        return self.concrete_access(self.access_O)

    def concrete_access_W(self):
        return self.concrete_access(self.access_W)

    def set_attr(self, key, value, overwrite=False):
        if key in self.attr and not overwrite:
            raise Exception(f"Key {key} already exists")
        self.attr[key] = value


class DataMovement(Operator):
    def __init__(self, domain, access_I, access_O, level, type_):
        assert type(access_I) in (
            isl.BasicMap,
            isl.Map,
            AccessRelation,
            TensorAccessRelation,
        )
        assert type(access_O) in (
            isl.BasicMap,
            isl.Map,
            AccessRelation,
            TensorAccessRelation,
        )

        self.domain = domain
        self.access_I = (
            AccessRelation(access_I)
            if type(access_I) in (isl.BasicMap, isl.Map)
            else access_I
        )
        self.access_O = (
            AccessRelation(access_O)
            if type(access_O) in (isl.BasicMap, isl.Map)
            else access_O
        )
        self.level = level

        assert type_ in ["I", "O", "W"]
        self.type_ = type_

    def convex_hull(self):
        return DataMovement(
            self.domain.convex_hull(),
            self.access_I.convex_hull(),
            self.access_O.convex_hull(),
            self.level,
            self.type_,
        )


class PartialSumDataMovement(DataMovement):
    def __init__(self, domain, domain_partial_sum, access_I, access_O, level, type_):
        super().__init__(domain, access_I, access_O, level, type_)
        self.domain_partial_sum = domain_partial_sum

    def convex_hull(self):
        return PartialSumDataMovement(
            self.domain.convex_hull(),
            self.domain_partial_sum.convex_hull(),
            self.access_I.convex_hull(),
            self.access_O.convex_hull(),
            self.level,
            self.type_,
        )


class DataMovementOperator:
    def __init__(
        self,
        domain,
        access_I,
        access_O,
        access_W,
        history_domains=None,
        history_schedules=None,
        data_movement=None,
        attr=None,
    ):
        if history_domains is None:
            history_domains = list()
        if history_schedules is None:
            history_schedules = list()
        if attr is None:
            attr = dict()

        assert type(domain) in (isl.BasicSet, isl.Set)
        assert type(access_I) in (
            isl.BasicMap,
            isl.Map,
            AccessRelation,
            TensorAccessRelation,
        )
        assert type(access_O) in (
            isl.BasicMap,
            isl.Map,
            AccessRelation,
            TensorAccessRelation,
        )
        assert type(access_W) in (
            isl.BasicMap,
            isl.Map,
            AccessRelation,
            TensorAccessRelation,
        )

        self.domain = domain
        self.access_I = (
            AccessRelation(access_I)
            if type(access_I) in (isl.BasicMap, isl.Map)
            else access_I
        )
        self.access_O = (
            AccessRelation(access_O)
            if type(access_O) in (isl.BasicMap, isl.Map)
            else access_O
        )
        self.access_W = (
            AccessRelation(access_W)
            if type(access_W) in (isl.BasicMap, isl.Map)
            else access_W
        )
        self.history_domains = history_domains
        self.history_schedules = history_schedules
        self.attr = attr

        if data_movement is None:
            self.data_movement = {"I": list(), "O": list(), "W": list()}
        else:
            assert isinstance(data_movement, dict)
            self.data_movement = data_movement

    def insert_buffer(self, buffer_name, _data_movement):
        assert buffer_name in ["I", "O", "W"]
        self.data_movement[buffer_name].append(_data_movement)

    def get_access_by_name(self, buffer_name):
        if buffer_name == "I":
            return self.access_I.offsets
        elif buffer_name == "O":
            return self.access_O.offsets
        elif buffer_name == "W":
            return self.access_W.offsets
        else:
            raise Exception(f"Unknown buffer name: {buffer_name}")

    def convex_hull(self):
        new_data_movement = {"I": list(), "O": list(), "W": list()}
        for name, data_movement in self.data_movement.items():
            for idx, dm in enumerate(data_movement):
                new_data_movement[name].append(dm.convex_hull())
        return DataMovementOperator(
            domain=self.domain.convex_hull(),
            access_I=self.access_I.convex_hull(),
            access_O=self.access_O.convex_hull(),
            access_W=self.access_W.convex_hull(),
            history_domains=[*self.history_domains, self.domain],
            history_schedules=[*self.history_schedules, "convex_hull"],
            data_movement=new_data_movement,
            attr={key: value for key, value in self.attr.items()},
        )

    def set_attr(self, key, value, overwrite=False):
        if key in self.attr and not overwrite:
            raise Exception(f"Key {key} already exists")
        self.attr[key] = value

    def get_attr(self, key):
        return self.attr[key]
