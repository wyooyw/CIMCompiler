from polycim.cli.common import show_args, to_abs_path
from polycim.config import get_config, set_raw_config_by_path
from polycim.op_compiler import parse_op_list, run_cimflow, run_polycim
from polycim.utils.logger import get_logger

logger = get_logger(__name__)


def parse_operator_args(subparsers):
    parser = subparsers.add_parser("op")
    parser.add_argument("--op-id", "-i", required=True, type=str, help="operator id")
    parser.add_argument(
        "--config-path", "-c", required=True, type=str, help="config path"
    )
    parser.add_argument(
        "--pimsim-cfg-path", "-p", type=str, default=None, help="pimsim config path"
    )
    parser.add_argument(
        "--output-path", "-o", required=True, type=str, help="output path"
    )
    parser.add_argument(
        "--data-movement-full-vectorize",
        action="store_true",
        help="data movement full vectorize",
    )
    parser.add_argument(
        "--polycim-disable-pretile", action="store_true", help="disable pretile"
    )
    parser.add_argument("--polycim-disable-affine", action="store_true", help="disable affine")
    parser.add_argument(
        "--polycim-disable-weight-rewrite", action="store_true", help="disable weight rewrite"
    )
    parser.add_argument(
        "--polycim-disable-second-stage", action="store_true", help="disable second stage"
    )
    parser.add_argument("--cimflow", action="store_true", help="run cimflow")
    parser.add_argument("--polycim", action="store_true", help="run polycim")
    parser.add_argument("--verify", action="store_true", help="verify")


def run_operator(args):
    args.output_path = to_abs_path(args.output_path)
    args.config_path = to_abs_path(args.config_path)
    set_raw_config_by_path(args.config_path)

    logger.info("Begin to compile operator.")
    logger.info(show_args(args))

    cim_cfg = get_config()

    pad_count = True
    delay_apply = True
    num_macros = cim_cfg.n_macro
    enable_weight_rewrite = True

    from polycim.exp.op_list import get_op_list

    op_list = get_op_list()
    op_list = {args.op_id: op_list[args.op_id]}

    # curr_time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # curr_time_str = curr_time_str + f"_{cim_cfg.n_comp}x{cim_cfg.n_group_vcol*8}"

    op = parse_op_list(op_list)
    # import pdb; pdb.set_trace()
    if args.polycim:
        run_polycim(args, cim_cfg, op)
    elif args.cimflow:
        run_cimflow(args, cim_cfg, op)
    else:
        assert False
