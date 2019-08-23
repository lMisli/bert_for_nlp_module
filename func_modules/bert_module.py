
import argparse
import logging
from scripts import common


parser = argparse.ArgumentParser("bert_module")
parser.add_argument("--language", type=str, default='En', help="Pre-training model used language, contains 'English/En', 'Multilingual/Multi','Chinses/Ch'. Default 'En'.")
parser.add_argument("--uncased", type=bool, default=True, help="Uncased means that the text has been lowercased before WordPiece tokenization,"
                                              " e.g., John Smith becomes john smith. The Uncased model also strips out any accent markers."
                                              " Cased means that the true case and accent markers are preserved. "
                                              "Typically, the Uncased model is better unless you know that case information is important for your task "
                                              "(e.g., Named Entity Recognition or Part-of-Speech tagging). Default'True'.")
parser.add_argument("--out_model_dir", type=str, help="Directory to save the path of selected pre-trained bert model.")

args = parser.parse_args()


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    # args.input_model_dir = "../bert/models"
    # args.out_model_file = "text"
    model_dir = None
    if args.language == "En" or args.language == "English":
        if args.uncased:
            model_dir = "/en_uncased"

        else:
            model_dir = "/en_cased"

    if args.language == "Multi" or args.language == "Multilingual":
        if args.uncased:
            model_dir ="/multi_uncased"
        else:
            model_dir = "/multi_cased"

    if args.language == "Ch" or args.language == "Chinese":
        model_dir = "/ch"

    out_model_file = common.generate_path(args.out_model_dir)
    with open(out_model_file, 'w', encoding='utf-8') as file:
        file.write(model_dir)
        logging.info('choosed bert saved into %s' % (out_model_file))






