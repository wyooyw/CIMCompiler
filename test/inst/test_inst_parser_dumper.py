import pytest
from simulator.inst import *
import json
import os

def show_diff(data1, data2):
    print(f"{len(data1)==len(data2)}")
    for line_idx, (code1, code2) in enumerate(zip(data1, data2)):
        print(f"{line_idx=}\n\t{code1=}\n\t{code2=}")

def test_parser_dumper(
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
            test_parser_dumper(parser1, dumper1, parser2, dumper2, path)
    
def test_legacy_parser_dumper(path):
    parser = LegacyParser()
    dumper = LegacyDumper()

    with open(path, 'r') as file:
        data = json.load(file)

    instructions = parser.parse(data)
    data2 = dumper.dump(instructions)
    assert data==data2

def test_cimflow_parser_dumper(path):
    parser = CIMFlowParser()
    dumper = CIMFlowDumper()

    with open(path, 'r') as file:
        data = json.load(file)

    instructions = parser.parse(data)
    data2 = dumper.dump(instructions)
    assert data==data2

def test_asm_parser_dumper(path):
    parser = AsmParser()
    dumper = AsmDumper()

    data = []
    with open(path, 'r') as file:
        for line in file.readlines():
            data.append(line.strip())

    instructions = parser.parse(data)
    data2 = dumper.dump(instructions)
    assert data==data2

if __name__=="__main__":
    test_all_parsers_dumpers()
    # this_dir = os.path.dirname(os.path.abspath(__file__))
    # test_legacy_parser_dumper(os.path.join(this_dir, "case1/legacy"))
    # test_asm_parser_dumper(os.path.join(this_dir, "case1/asm"))
    # test_cimflow_parser_dumper(os.path.join(this_dir, "case1/cimflow"))
    # test_legacy_asm_parser_dumper(os.path.join(this_dir, "case1/legacy"))
    # parser = LegacyParser()
    # dumper = CIMFlowDumper()
    
    # insts = parser.parse_file("/home/wangyiou/project/cim_compiler_frontend/playground/test/inst/case1/legacy")
    # dumper.dump_to_file(insts, "/home/wangyiou/project/cim_compiler_frontend/playground/test/inst/case1/cimflow")