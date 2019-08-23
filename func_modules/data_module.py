
import argparse
from scripts import dataprocess, common
import logging

parser = argparse.ArgumentParser("data_module")
parser.add_argument("--input_data", type=str, help="Data path to load.")
parser.add_argument("--output_dir", type=str, help="Data path to save.")
args = parser.parse_args()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    #args.input_data = 'D:/corpus/toxic comment/jigsaw-toxic-comment-classification-challenge/cleaned_train_part.csv'
    #args.output_data = 'test'
    data = dataprocess.load_data(args.input_data)
    #output_file = os.path.join(args.output_dir, "dataset.csv")
    output_predict_file = common.generate_path(args.output_dir)
    data.to_csv(output_predict_file,index=0)
    logging.info('Load data to %s' %(output_predict_file))