import copy
import time

from polycim.op.base_operator import Operator
from polycim.utils.logger import get_logger

logger = get_logger(__name__)


class Pass:
    def __init__(self):
        pass

    def apply(self, operator):
        raise NotImplementedError


class DepthFirstPass(Pass):
    def __init__(
        self,
        fix_schedule=None,
        schedule_as_key=False,
    ):
        super().__init__()
        self.fix_schedule = fix_schedule
        self.schedule_as_key = schedule_as_key

    def apply(self, operator):
        raise NotImplementedError


class BreadthFirstPass(Pass):
    def __init__(self):
        super().__init__()

    def apply(self, operator):
        raise NotImplementedError

    def apply_all(self):
        raise NotImplementedError

    def get_result(self):
        raise NotImplementedError


class EndBreadthFirstPass(BreadthFirstPass):
    def __init__(self):
        super().__init__()
        self.result_list = list()

    def apply(self, operator):
        self.result_list.append(operator)

    def apply_all(self):
        pass

    def get_result(self):
        return self.result_list


class Schedule:
    def __init__(self):
        pass

    def dumps(self):
        raise NotImplementedError


class SchedulePassResult:
    def __init__(self, op: Operator, schedule: Schedule):
        self.op = op
        self.schedule = schedule


class ScheduleList:
    def __init__(self, schedule_list=list()):
        self.schedule_list = schedule_list

    def add_schedule(self, schedule):

        new_schedule_list = []
        for sch in self.schedule_list:
            new_schedule_list.append(copy.deepcopy(sch))
        new_schedule_list.append(copy.deepcopy(schedule))
        return ScheduleList(new_schedule_list)

    def __hash__(self):
        return hash(tuple(hash(schedule) for schedule in self.schedule_list))

    def __eq__(self, other):
        return self.schedule_list == other.schedule_list


class PassManager:
    def __init__(self, pass_list):
        self.pass_list = pass_list
        self.check_pass_list()
        self.time_per_pass = dict()

    def get_num_ops(self):
        assert isinstance(self.pass_list[-1], EndBreadthFirstPass)
        return len(self.pass_list[-1].get_result())

    def check_pass_list(self):
        self.pass_list.append(EndBreadthFirstPass())
        # for pass_ in self.pass_list[:-1]:
        #     if not isinstance(pass_, DepthFirstPass):
        #         raise ValueError(f"Invalid pass type: {type(pass_)}")
        # if not isinstance(self.pass_list[-1], BreadthFirstPass):
        # raise ValueError(f"Invalid pass type: {type(self.pass_list[-1])}")

    def update_result(self, result):
        raise NotImplementedError

    def _apply_pass(self, op, pass_):
        start_time = time.time()
        result = pass_.apply(op)
        end_time = time.time()

        pass_name = pass_.__class__.__name__
        self.time_per_pass[pass_name] = self.time_per_pass.get(pass_name, 0) + (
            end_time - start_time
        )
        return result

    def _apply_all_pass(self, pass_):
        start_time = time.time()
        pass_.apply_all()
        end_time = time.time()

        pass_name = pass_.__class__.__name__
        self.time_per_pass[pass_name] = self.time_per_pass.get(pass_name, 0) + (
            end_time - start_time
        )

    def get_time_per_pass(self, sort_by_time=True):
        if sort_by_time:
            # sort by time
            sorted_time_per_pass = sorted(
                self.time_per_pass.items(), key=lambda x: x[1], reverse=True
            )
        else:
            # sort by pass_list
            sorted_time_per_pass = sorted(
                self.time_per_pass.items(),
                key=lambda x: self.pass_list.index(x[0]),
                reverse=False,
            )
        return sorted_time_per_pass

    def show_time_per_pass(self, sort_by_time=True):
        sorted_time_per_pass = self.get_time_per_pass(sort_by_time)

        # add total time and percentage
        total_time = sum(self.time_per_pass.values())
        s = f"Total time: {total_time:.2f}s"
        s += "Time per pass: "
        for pass_name, time_ in sorted_time_per_pass:
            s += f"\t{pass_name}: {time_:.2f}s ({time_/total_time*100:.2f}%)"
        logger.info(s)

    def _apply_until_breadth(self, op, step=0):
        pass_ = self.pass_list[step]
        if isinstance(pass_, DepthFirstPass):
            for result in self._apply_pass(op, pass_):
                new_op, schedule = result.op, result.schedule
                if pass_.schedule_as_key:
                    new_op.attr["PassManager::schedule_keys"] = new_op.attr[
                        "PassManager::schedule_keys"
                    ].add_schedule(schedule)
                self._apply_until_breadth(new_op, step + 1)
        elif isinstance(pass_, BreadthFirstPass):
            self._apply_pass(op, pass_)
            return

    def _apply_op_list_until_breadth(self, op_list, step=0):
        for op in op_list:
            self._apply_until_breadth(op, step)

    def apply(self, op):
        op.attr["PassManager::schedule_keys"] = ScheduleList()
        breadth_pass_indices = [
            i
            for i, pass_ in enumerate(self.pass_list)
            if isinstance(pass_, BreadthFirstPass)
        ]
        if breadth_pass_indices[0] != 0:
            breadth_pass_indices.insert(0, -1)

        op_list = [op]
        for i in range(1, len(breadth_pass_indices)):
            begin_pass_id = breadth_pass_indices[i - 1] + 1
            stop_pass_id = breadth_pass_indices[i]
            self._apply_op_list_until_breadth(op_list, begin_pass_id)
            stop_pass = self.pass_list[stop_pass_id]
            self._apply_all_pass(stop_pass)
            op_list = stop_pass.get_result()

        return op_list
