import pytest
from simulator.inst import *
import json
import os

def show_diff(data1, data2):
    print(f"{len(data1)==len(data2)}")
    for line_idx, (code1, code2) in enumerate(zip(data1, data2)):
        print(f"{line_idx=}\n\t{code1=}\n\t{code2=}")

def _test_parser_dumper(
        parser1, 
        dumper1, 
        parser2, 
        dumper2, 
        path):

    # Parse the original data
    original_data, insts1 = parser1.parse_file(path)
    
    # Dump the instructions back to data
    dumped_data = dumper2.dump(insts1)
    
    # Parse the dumped data again
    insts2 = parser2.parse(dumped_data)
    
    # Dump again to compare with the original
    final_data = dumper1.dump(insts2)
    
    if original_data != final_data:
        show_diff(original_data, final_data)
        exit()
    assert original_data == final_data, "Mismatch between original and final data"

@pytest.mark.parametrize("parser_class, dumper_class, path", [
    (LegacyParser, LegacyDumper, "legacy"),
    (CIMFlowParser, CIMFlowDumper, "cimflow"),
    (AsmParser, AsmDumper, "asm"),
])
def test_individual_parser_dumper(parser_class, dumper_class, path):
    parser = parser_class()
    dumper = dumper_class()
    this_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(this_dir, "case1", path)

    with open(path, 'r') as file:
        if path.endswith("asm"):
            data = [line.strip() for line in file.readlines()]
        else:
            data = json.load(file)

    instructions = parser.parse(data)
    data2 = dumper.dump(instructions)
    assert data == data2

def test_all_parsers_dumpers():
    parser_classes = [LegacyParser, AsmParser, CIMFlowParser]
    dumper_classes = [LegacyDumper, AsmDumper, CIMFlowDumper]
    case_paths = ["legacy","asm", "cimflow"]
    n_class = len(parser_classes)
    assert len(parser_classes)==len(dumper_classes)

    this_dir = os.path.dirname(os.path.abspath(__file__))

    for i in range(n_class):
        for j in range(n_class):
            parser1 = parser_classes[i]()
            parser2 = parser_classes[j]()
            dumper1 = dumper_classes[i]()
            dumper2 = dumper_classes[j]()
            path = os.path.join(this_dir, "case1", case_paths[i])
            _test_parser_dumper(parser1, dumper1, parser2, dumper2, path)

if __name__=="__main__":
    test_all_parsers_dumpers()