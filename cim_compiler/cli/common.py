import os
from cim_compiler.simulator.inst import (
    LegacyParser,
    AsmParser,
    CIMFlowParser
)
from cim_compiler.utils.logger import get_logger
logger = get_logger(__name__)

def to_abs_path(path, parent=os.getcwd()):
    if path is None:
        return None
    if not os.path.isabs(path):
        return os.path.join(parent, path)
    return path

def show_args(args):
    s = "Arguments:\n"
    for key, value in vars(args).items():
        s += f"  {key}: {value}\n"
    return s

def uniform_parse_code(code_type, code_file_path):
    parser_classes = {
        "legacy": LegacyParser,
        "asm": AsmParser,
        "cimflow": CIMFlowParser
    }

    if code_type == "any":
        for code_type, parser_cls in parser_classes.items():
            try:
                _, data = parser_cls().parse_file(code_file_path)
                logger.info(f"Parse {code_file_path} success with {parser_cls.__name__}.")
                return data, code_type
            except Exception as e:
                pass
        raise ValueError("Failed to parse source file.")
    else:
        parser = parser_classes[code_type]()
        _, data = parser.parse_file(code_file_path)
        return data, code_type