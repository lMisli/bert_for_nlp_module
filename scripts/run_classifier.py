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

import os
import tensorflow as tf
from scripts import common, modeling, tokenization, dataprocess
import argparse
from tensorflow.python.distribute.cross_device_ops import AllReduceCrossDeviceOps
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.client import device_lib

parser = argparse.ArgumentParser("run_classifier")
parser.add_argument("--train_data", type=str, help="train data path")
parser.add_argument("--output_dir", type=str, help="trained model and results directory")
parser.add_argument("--add_layer", type=str, default="multi_label_classifier", help="Name of added layers after BERT")
parser.add_argument("--train_column_names", type=str, help="Content columns (X), separated by ' '. E.g., 'col1 col2,'")
parser.add_argument("--label_column_names", type=str, help="Label columns used to learn (Y), separated by ' '. E.g., 'col1 col2,'")
#parser.add_argument("--pretrained_dir", type=str, help="pretrained model directory")
parser.add_argument("--bert_config_file",type=str, default='../bert/uncased_L-12_H-768_A-12/bert_config.json', help="config.json file path")
parser.add_argument("--vocab_file", type=str, default='../bert/uncased_L-12_H-768_A-12/vocab.txt', help="vocab file path")
parser.add_argument("--init_checkpoint_file", type=str, default='../bert/uncased_L-12_H-768_A-12/bert_model.ckpt', help="init checkponit file, default is the bert_model init checkpoint")
#parser.add_argument("--data_tfrecord_dir", type=str, default='data_records', help="directory containing data tf_record file")

parser.add_argument("--do_lower_case", type=bool, default=True, help="whether to convert words to lowercase")
parser.add_argument("--is_training_bert", type=bool, default=False, help="whether to train bert model")
parser.add_argument("--max_seq_length",type=int, default=512,  help="The maximum total input sequence length after WordPiece tokenization."
                         "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Total number of training epochs to perform.")
parser.add_argument("--train_batch_size", type=int, default=16, help="Total batch size for training.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help= "The initial learning rate for Adam.")

parser.add_argument("--warmup_proportion", type=float, default=0.1, help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
parser.add_argument("--save_checkpoints_steps", type=int, default=10000, help="How often to save the model checkpoint.")
parser.add_argument("--iterations_per_loop", type=int, default=10000, help= "How many steps to make in each estimator call.")

parser.add_argument("--use_gpu", type=bool, default=False, help="Whether to use GPU. when True the 'use_tpu' is setted False. ")
parser.add_argument("--num_gpu_cores", type=int, help="Only used if `use_gpu` is True. Total number of GPU cores to use. defaut use all avialble GPUs")
parser.add_argument("--use_fp16", type=bool, default=False, help="Whether to use fp16.")

parser.add_argument("--use_tpu", type=bool, default=False, help="Whether to use TPU or GPU/CPU.")
parser.add_argument("--tpu_name", type=str, help="The Cloud TPU to use for training. "
                                                 "This should be either the name used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")
parser.add_argument("--tpu_zone", type=str, help="[Optional] GCE zone where the Cloud TPU is located in. "
                                                 "If not specified, we will attempt to automatically detect the GCE project from metadata.")
parser.add_argument("--master", type=str, help="[Optional] TensorFlow master URL.")
parser.add_argument("--num_tpu_cores", type=int, default=8, help="Only used if `use_tpu` is True. Total number of TPU cores to use.")
parser.add_argument("--gcp_project", type=str, help="[Optional] Project name for the Cloud TPU-enabled project. "
                                                    "If not specified, we will attempt to automatically detect the GCE project from metadata.")
#parser.add_argument("--bert_model", type=str, default="uncased_L-12_H-768_A-12", help="Model config name")



args = parser.parse_args()





def train():
  tf.logging.set_verbosity(tf.logging.INFO)

  df = dataprocess.load_data(args.train_data)
  train_column_names = args.train_column_names.split(' ')
  label_column_names = args.label_column_names.split(' ')
  label_len = len(label_column_names)

  tokenization.validate_case_matches_checkpoint(args.do_lower_case,
                                                args.init_checkpoint_file)


  bert_config = modeling.BertConfig.from_json_file(args.bert_config_file)

  if args.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (args.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(args.output_dir)
  processor = common.toxicCommentProcessor()
  #vocab_file = os.path.join(args.bert_model,"vocab.txt")
  tokenizer = tokenization.FullTokenizer(
      vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)

  tpu_cluster_resolver = None
  if args.use_tpu and args.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        args.tpu_name, zone=args.tpu_zone, project=args.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2

  #num_gpu_cores = 0
  if args.use_gpu and args.num_gpu_cores == None:
      num_gpu_cores = len([x for x in device_lib.list_local_devices() if x.device_type=='GPU'])
  else:
      num_gpu_cores = args.num_gpu_cores
  if args.use_gpu and int(num_gpu_cores) >= 2:
      tf.logging.info("Use normal RunConfig， GPU number: %", num_gpu_cores)
      dist_strategy = tf.contrib.distribute.MirroredStrategy(
          num_gpus=num_gpu_cores,
          cross_device_ops=AllReduceCrossDeviceOps('nccl', num_packs=num_gpu_cores)
      )
      log_every_n_steps = 8
      run_config = RunConfig(
          train_distribute=dist_strategy,
          eval_distribute=dist_strategy,
          log_step_count_steps=log_every_n_steps,
          model_dir=args.output_dir,
          save_checkpoints_steps=args.save_checkpoints_steps)

  else:
      tf.logging.info("Use TPURunConfig")
      run_config = tf.contrib.tpu.RunConfig(
          cluster=tpu_cluster_resolver,
          master=args.master,
          model_dir=args.output_dir,
          save_checkpoints_steps=args.save_checkpoints_steps,
          tpu_config=tf.contrib.tpu.TPUConfig(
              iterations_per_loop=args.iterations_per_loop,
              num_shards=args.num_tpu_cores,
              per_host_input_for_training=is_per_host))


  train_examples = processor.get_train_examples(df, train_column_names, label_column_names)
  num_train_steps = int(len(train_examples) / args.train_batch_size * args.num_train_epochs)
  num_warmup_steps = int(num_train_steps * args.warmup_proportion)

  model = common.get_model(args.add_layer)

  model_fn = common.model_fn_builder(
      bert_config=bert_config,
      is_training_bert=args.is_training_bert,
      num_labels=len(label_column_names),
      init_checkpoint=args.init_checkpoint_file,
      learning_rate=args.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=args.use_tpu,
      use_one_hot_embeddings=args.use_tpu,
      use_gpu=args.use_gpu,
      num_gpu_cores=num_gpu_cores,
      fp16=args.use_fp16,
      model=model)


  if args.use_gpu and int(num_gpu_cores) >= 2:
      tf.logging.info("Use normal Estimator")
      estimator = Estimator(
          model_fn=model_fn,
          params={},
          config=run_config
      )

  else:
      tf.logging.info("Use TPUEstimator")
      # If TPU is not available, this will fall back to normal Estimator on CPU
      # or GPU.
      estimator = tf.contrib.tpu.TPUEstimator(
          use_tpu=args.use_tpu,
          model_fn=model_fn,
          config=run_config,
          train_batch_size=args.train_batch_size)

  train_file = os.path.join(args.output_dir, "train.tf_record")
  if not os.path.isfile(train_file):
    common.file_based_convert_examples_to_features(
        train_examples, args.max_seq_length, tokenizer, train_file)
  tf.logging.info("***** Running training *****")
  tf.logging.info("  Num examples = %d", len(train_examples))
  tf.logging.info("  Batch size = %d", args.train_batch_size)
  tf.logging.info("  Num steps = %d", num_train_steps)
  train_input_fn = common.file_based_input_fn_builder(
    input_file=train_file,
    seq_length=args.max_seq_length,
    label_length=label_len,
    is_training=True,
    drop_remainder=True,
    batch_size=args.train_batch_size)

  tensors_to_log = {"train loss": "loss/Mean:0"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=1)
  estimator.train(input_fn=train_input_fn, hooks=[logging_hook], max_steps=num_train_steps)




if __name__ == "__main__":

  data_path = args.train_data
  #train_data = dataprocess.load_data(data_path)
  args.output_dir = "../lstm_output"
  args.train_data = "D:/corpus/toxic comment/jigsaw-toxic-comment-classification-challenge/part/train_16.csv"
  args.train_column_names = "comment_text"
  args.label_column_names = "toxic severe_toxic obscene threat insult identity_hate"
  args.add_layer = "lstm_multi_label_classifier"

  # #train_data = dataprocess.load_data(
  #     "D:/corpus/toxic comment/jigsaw-toxic-comment-classification-challenge/cleaned_traine.csv")

  #config = common.model_config(config_dir="bert", output_dir="output_dir1")

  train()



