
import argparse
import logging
#from scripts import common
import zipfile
import urllib.request as request
import os
import shutil


parser = argparse.ArgumentParser("bert_module")
parser.add_argument("--language", type=str, default='En', help="Pre-training model used language, contains 'English/En', 'Multilingual/Multi','Chinses/Ch'. Default 'En'.")
parser.add_argument("--uncased", type=bool, default=True, help="Uncased means that the text has been lowercased before WordPiece tokenization,"
                                              " e.g., John Smith becomes john smith. The Uncased model also strips out any accent markers."
                                              " Cased means that the true case and accent markers are preserved. "
                                              "Typically, the Uncased model is better unless you know that case information is important for your task "
                                              "(e.g., Named Entity Recognition or Part-of-Speech tagging). Default'True'.")
parser.add_argument("--out_model_dir", type=str, help="Directory to save the pre-trained bert model.")

args = parser.parse_args()


en_uncased_url = "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip"
en_cased_url = "https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip"

multi_uncased_url = "https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip"
multi_cased_url = "https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip"
ch_url = "https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip"

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    # args.out_model_dir = "models"
    if not os.path.exists(args.out_model_dir):
        os.makedirs(args.out_model_dir)
    filename = "model.zip"
    filepath = os.path.join(args.out_model_dir, filename)
    logging.info("Downloading pre-trained BERT weights...")
    if args.language == "En" or args.language == "English":
        if args.uncased:
            filepath, _ = request.urlretrieve(en_uncased_url, filepath)

        else:
            filepath, _ = request.urlretrieve(en_cased_url, filepath)

    if args.language == "Multi" or args.language == "Multilingual":
        if args.uncased:
            filepath, _ = request.urlretrieve(multi_uncased_url, filepath)
        else:
            filepath, _ = request.urlretrieve(multi_cased_url, filepath)

    if args.language == "Ch" or args.language == "Chinese":
        filepath, _ = request.urlretrieve(ch_url, filepath)

    # unzip
    logging.info("Extract BERT files...")
    with zipfile.ZipFile(filepath, 'r') as zip:
        zip.extractall(args.out_model_dir)

    # remove zipfile
    if os.path.exists(filepath):
        os.remove(filepath)


    for root, dirs, files in os.walk(args.out_model_dir, topdown=False):
        for file in files:
            try:
                shutil.move(os.path.join(root,file), args.out_model_dir)
            except OSError:
                pass






