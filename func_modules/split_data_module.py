import argparse
import logging
from scripts import dataprocess, common

parser = argparse.ArgumentParser("split_data_module")
parser.add_argument("--input_dir", type=str, help="Data path to load.")
parser.add_argument("--row_split_rate", type=float, default=0.7, help="When split data by rows, the propotion of the first data part.")
parser.add_argument("--random", type=bool, default=True, help="Whether to split data randomly.")
parser.add_argument("--random_seed", type=int, default=200, help="When random is true, it works. Default 200.")
parser.add_argument("--output_dir1", type=str, help="Split data part1 path to save.")
parser.add_argument("--output_dir2", type=str, help="Split data part2 path to save.")

args = parser.parse_args()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    input_data_file = common.parse_path(args.input_dir)
    output1 = common.generate_path(args.output_dir1)
    output2 = common.generate_path(args.output_dir2)
    dataprocess.split_data_by_rows(input_path=input_data_file,
                                   output_path1=output1,
                                   output_path2=output2,
                                   split_rate=args.row_split_rate,
                                   random=args.random,
                                   random_seed=args.random_seed)
    logging.info('Saved splited part-1 data to %s'%(output1))
    logging.info('Saved splited part-2 data to %s' % (output2))

