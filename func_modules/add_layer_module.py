import argparse
import json
import logging
from scripts import common

parser = argparse.ArgumentParser("custom_add_layer")
parser.add_argument("--output_dir", type=str, help="Directory to save custom added layer config file.")
parser.add_argument("--layer_name", type=str, help="Class to choose, contains 'regressor', 'multi-label-classifier', 'multi-class-classifier','binary-classifier'.")

args = parser.parse_args()


if __name__ == "__main__":
    # args.output_path = "text.txt"
    # args.label_num = 6
    # args.category ="multi-label-classifier"
    logging.getLogger().setLevel(logging.INFO)
    config = {
        "layer_name": args.layer_name
    }

    output_file = common.generate_path(args.output_dir)
    with open(output_file,'w',encoding='utf-8') as outf:
        json.dump(config, outf)
    logging.info("added layer saved to path: %s", output_file)




