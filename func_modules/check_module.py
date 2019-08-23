# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

from scripts import  modeling

import argparse



parser = argparse.ArgumentParser("check")
parser.add_argument("--input", type=str, help="train data path")





args = parser.parse_args()





def train():
    bert_config = modeling.BertConfig.from_json_file(args.input)
    print(bert_config)


if __name__ == "__main__":

  # data_path = args.train_data
  # #train_data = dataprocess.load_data(data_path)
  # args.train_column_names = "comment_text"
  # args.label_column_names = "toxic, severe_toxic, obscene, threat, insult, identity_hate"
  #
  # # #train_data = dataprocess.load_data(
  # #     "D:/corpus/toxic comment/jigsaw-toxic-comment-classification-challenge/cleaned_traine.csv")
  #
  # #config = common.model_config(config_dir="bert", output_dir="output_dir1")

    train()



