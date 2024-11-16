from simulator.simulator import Memory, MemorySpace, Simulator, SpecialReg
import argparse
import os
import json

def parse_args():
    parser = argparse.ArgumentParser(description='Simulate and compute statistics for a given model')
    parser.add_argument('-i', '--input', type=str, help='Code to run')
    parser.add_argument('-c', '--config', type=str, help='Config')
    parser.add_argument('-o', '--output', type=str, help='Output file to save the results')
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.input, "r") as f:
        code = json.load(f)

    simulator = Simulator.from_config(args.config)
    status, stats, flat = simulator.run_code(
        code, 
        record_flat=False
    )
    assert status == simulator.FINISH

    stats.dump(args.output)

if __name__=="__main__":
    main()