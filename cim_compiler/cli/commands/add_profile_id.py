
from cim_compiler.utils.logger import get_logger
from cim_compiler.cli.common import show_args, to_abs_path
import json
from cim_compiler.utils.json_utils import (
    dumps_list_of_dict,
    dumps_dict_of_list_of_dict
)

logger = get_logger(__name__)

def parse_convert_args(subparsers):
    parser = subparsers.add_parser('convert')
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, required=True)
    parser.add_argument("--profile-id", "-p", type=str, required=True)


def run_convert(args):
    args.input = to_abs_path(args.input)
    args.output = to_abs_path(args.output)

    logger.info("Begin to add profile id.")
    logger.info(show_args(args))

    with open(args.input, "r") as f:
        data = json.load(f)
    
    if isinstance(data, list):
        assert all(isinstance(item, dict) for item in data)
        for inst in data:
            inst["profile_id"] = args.profile_id
        save_text = dumps_list_of_dict(data)
    elif isinstance(data, dict):
        assert all(isinstance(item, list) for item in data)
        for core_id, core_inst_list in data:
            for inst in core_inst_list:
                inst["profile_id"] = args.profile_id
        save_text = dumps_dict_of_list_of_dict(data)
    else:
        assert False

    with open(args.output, "w") as f:
        f.write(save_text)
        
    logger.info("Convert done.")
