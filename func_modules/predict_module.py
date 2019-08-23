
"""BERT finetuning runner."""


import os
import tensorflow as tf
import pandas as pd
from scripts import common, modeling, tokenization, dataprocess

import numpy as np
import argparse
from tensorflow.python.distribute.cross_device_ops import AllReduceCrossDeviceOps
from tensorflow.python.estimator.run_config import RunConfig
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.client import device_lib

parser = argparse.ArgumentParser("run_classifier")
parser.add_argument("--test_data", type=str, help="Test data path.")
parser.add_argument("--bert_dir", type=str, help="Directory contains all kinds of pre-trained bert.")
parser.add_argument("--output_dir", type=str, help="Directory to save predicted results.")
parser.add_argument("--bert_model", type=str,  help="Config file of selected bert model.")
parser.add_argument("--added_layer_config", type=str,  help="Config file of added layers after BERT.")
parser.add_argument("--predict_column_names", type=str, help="Colunms name used to predict,separated by ' ', such as 'col1 col2'.")
parser.add_argument("--trained_model_dir", type=str, help="Directory saved the trained model.")
parser.add_argument("--do_lower_case", type=bool, default=True, help="Whether to convert words to lowercase. Should be True for uncased models and False for cased models.")
parser.add_argument("--max_seq_length",type=int, default=512,  help="The maximum total input sequence length after WordPiece tokenization."
                         "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--predict_batch_size", type=int, default=4, help="Batch size for predicting. Batch size for each GPU when num_gpu_cores >=2.")
parser.add_argument("--iterations_per_loop", type=int, default=10000, help= "How many steps to make in each estimator call.")
parser.add_argument("--use_gpu", type=bool, default=True, help="Whether to use GPU. when True the 'use_tpu' is setted False. ")
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

args = parser.parse_args()


def predict():
  #FLAGS = common.model_config()
  tf.logging.set_verbosity(tf.logging.INFO)

  args.test_data = common.parse_path(args.test_data)
  args.bert_model = common.parse_path(args.bert_model)
  args.added_layer_config = common.parse_path(args.added_layer_config)

  df = dataprocess.load_data(args.test_data)
  test_column_names = args.predict_column_names.split(' ')

  ckpt = tf.train.get_checkpoint_state(args.trained_model_dir)
  checkpoint_file = ckpt.model_checkpoint_path

  tokenization.validate_case_matches_checkpoint(args.do_lower_case,
                                                checkpoint_file)

  file = open(args.bert_model, 'r', encoding='utf-8')
  sub_dir = file.read().strip('\n')
  file.close()
  bert_model_dir = args.bert_dir + sub_dir

  bert_config_file = os.path.join(bert_model_dir, "bert_config.json")
  bert_config = modeling.BertConfig.from_json_file(bert_config_file)

  if args.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (args.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(args.output_dir)
  processor = common.toxicCommentProcessor()

  vocab_file = os.path.join(bert_model_dir, "vocab.txt")
  tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=args.do_lower_case)

  tpu_cluster_resolver = None
  if args.use_tpu and args.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        args.tpu_name, zone=args.tpu_zone, project=args.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  if args.use_gpu and args.num_gpu_cores == None:
      num_gpu_cores = len([x for x in device_lib.list_local_devices() if x.device_type == 'GPU'])
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
          log_step_count_steps=log_every_n_steps)

  else:
      tf.logging.info("Use TPURunConfig")
      run_config = tf.contrib.tpu.RunConfig(
          cluster=tpu_cluster_resolver,
          master=args.master,
          tpu_config=tf.contrib.tpu.TPUConfig(
              iterations_per_loop=args.iterations_per_loop,
              num_shards=args.num_tpu_cores,
              per_host_input_for_training=is_per_host))

  num_train_steps = None
  num_warmup_steps = None
  learning_rate = None
  added_layer = common.loadJsonConfig(args.added_layer_config)

  model = common.get_model(added_layer['layer_name'])
  # model = common.get_model(args.add_layer)
  label_num = added_layer['label_num']
  model_fn = common.model_fn_builder(
      bert_config=bert_config,
      is_training_bert = False,
      num_labels=label_num,
      init_checkpoint=checkpoint_file,
      learning_rate=learning_rate,
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
          predict_batch_size=args.predict_batch_size)

  predict_examples = processor.get_test_examples(df, test_column_names, label_num)
  num_actual_predict_examples = len(predict_examples)
  if args.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % args.predict_batch_size != 0:
          predict_examples.append(common.PaddingInputExample())

  predict_file = os.path.join(args.output_dir, "predict.tf_record")
  if not os.path.isfile(predict_file):
      common.file_based_convert_examples_to_features(predict_examples, args.max_seq_length, tokenizer, predict_file)

  tf.logging.info("***** Running prediction*****")
  tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                  len(predict_examples), num_actual_predict_examples,
                  len(predict_examples) - num_actual_predict_examples)
  tf.logging.info("  Batch size = %d", args.predict_batch_size)

  predict_drop_remainder = True if args.use_tpu else False
  predict_input_fn = common.file_based_input_fn_builder(
      input_file=predict_file,
      seq_length=args.max_seq_length,
      label_length= label_num,
      is_training=False,
      drop_remainder=predict_drop_remainder,
      batch_size= args.predict_batch_size)

  result = estimator.predict(input_fn=predict_input_fn)


  #output_predict_file = os.path.join(args.output_dir, "predict_results.csv")
  output_predict_file = common.generate_path(args.output_dir)
  #with tf.gfile.GFile(output_predict_file, "w") as writer:
  num_written_lines = 0
  tf.logging.info("***** Predict results *****")
  predict_res = []
  for (i, prediction) in enumerate(result):
      probabilities = prediction["probabilities"]
      neg_preds = np.zeros(shape=probabilities.shape, dtype=float)
      pos_preds = np.ones(shape=probabilities.shape, dtype=float)
      predictions = np.where(probabilities < 0.5, neg_preds, pos_preds)
      if i >= num_actual_predict_examples:
          break
      piece = np.r_[probabilities, predictions]
      predict_res.append(piece)
      num_written_lines += 1
  output_colums = []
  for i in range(len(predict_res[0])//2):
      col_name = "probability_" + str(i+1)
      output_colums.append(col_name)
  for i in range(len(predict_res[0])//2):
      col_name = "prediction_" + str(i+1)
      output_colums.append(col_name)
  out_df = pd.DataFrame(columns=output_colums, data=predict_res)
  print(out_df.head(3))
  out_df = pd.concat([df, out_df], axis = 1)
  print(out_df.head(3))
  out_df.to_csv(output_predict_file,index=False)





if __name__ == "__main__":
  #test_data = dataprocess.load_data(
  #      "D:\\corpus\\toxic comment\\jigsaw-toxic-comment-classification-challenge\\cleaned_test_data_part.csv")
  # test_data = dataprocess.load_data(args.test_data)
  # test_column_names = ["comment_text"]
  #config = common.model_config(config_dir=args.pretrained_dir, output_dir=args.output_dir)
  #args.output_dir = "output_dir_part"
  # args.use_gpu = False
  #   # args.init_checkpoint_file = "output_dir/model.ckpt-15294"
  #   # args.label_num = 6

  predict()