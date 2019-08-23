
import collections
from scripts import custom_optimization, modeling, optimization, tokenization, custom_layer
import tensorflow as tf
import pandas as pd
import numpy as np
import json
import os


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class toxicCommentProcessor(object):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, df:pd.DataFrame, train_column_names:list, label_column_names: list):
    """See base class."""
    # lines = self._read_tsv(
    #     os.path.join(data_dir, "multinli",
    #                  "multinli.train.%s.tsv" % self.language))
    examples = []
    tf.logging.info("loading data ...")
    for (i, line) in df.iterrows():
      label = line[label_column_names]
      label = label.values
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line.loc[train_column_names].values[0])
      #label = tokenization.convert_to_unicode(line.loc[label_column_names].values)
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    tf.logging.info("loading data finished...")
    return examples

  def get_dev_examples(self, df:pd.DataFrame, train_column_names:list, label_column_names: list):
    """See base class."""
    examples = []
    for (i, line) in enumerate(df):
        if i == 0:
            continue
        guid = "dev-%d" % (i)
        text_a = tokenization.convert_to_unicode(line[train_column_names])
        label = tokenization.convert_to_unicode(line[label_column_names])
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

  def get_test_examples(self, df:pd.DataFrame, test_column_names:list, label_num:int):
    """See base class."""
    examples = []
    #label = [0 for i in range(labels_num)]
    for (i, line) in df.iterrows():
        guid = "test-%d" % (i)
        #print(line.loc[test_column_names].values[0])
        text_a = tokenization.convert_to_unicode(line.loc[test_column_names].values[0])
        label = np.array([0 for i in range(label_num)])
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples

  def get_labels(self, labels):
    """See base class."""
    return labels.tolist()


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


def convert_single_example(ex_index, example,  max_seq_length, tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  # label_map = {}
  # for (i, label) in enumerate(label_list):
  #   label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length


  label = example.label.tolist()
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    if label != None:
        tf.logging.info("label: %s " % " ".join(str(x) for x in label))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples,  max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    if feature.label_id == None:
        features["label_ids"] = None
    else:
        features["label_ids"] = create_int_feature(feature.label_id)
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, label_length, is_training,
                                drop_remainder, batch_size):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([label_length], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    #batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def get_model(layer_name):

    def multi_label_classifier(bert_config, is_training_bert, is_training, input_ids, input_mask, segment_ids,
                     labels, num_labels, use_one_hot_embeddings):
        """Creates the multi-label classification layers."""
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training_bert,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
         )

        output_layer = model.get_pooled_output()  # get the senquence-level output

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())


        with tf.variable_scope("loss"):
            if is_training_bert:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            probabilities = tf.nn.sigmoid(logits, name="probability")

            per_example_loss = None
            loss = None
            if is_training:
                labels = tf.cast(labels,dtype=tf.float32)
                per_example_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name="loss")
                loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, logits, probabilities)

    def lstm_multi_label_classifier(bert_config, is_training_bert, is_training, input_ids, input_mask, segment_ids,
                     labels, num_labels, use_one_hot_embeddings):
        model = modeling.BertModel(
            config=bert_config,
            is_training=is_training_bert,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings
        )
        output_layer = model.get_sequence_output()
        loss_func = tf.nn.sigmoid_cross_entropy_with_logits
        lstm_out = custom_layer.custom_layer(input=output_layer, is_training=is_training, num_units=128, num_layers=1).lstm()
        logits, probabilities = custom_layer.custom_layer(input=lstm_out, label=labels, is_training=is_training).dense()
        loss, per_example_loss = custom_layer.custom_layer(input=logits, is_training=is_training, label=labels, loss_func=loss_func).get_loss_func()

        return (loss, per_example_loss, logits, probabilities)

    if layer_name == "multi_label_classifier":
        return multi_label_classifier
    elif layer_name == "lstm_multi_label_classifier":
        return lstm_multi_label_classifier
    elif layer_name == "binary_classifier":
        return None
    elif layer_name == "multi_class_classifier":
        return None



def get_module(module_name):
    pass



def model_fn_builder(bert_config,is_training_bert, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, use_gpu, num_gpu_cores, fp16, model):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = model(
        bert_config, is_training_bert,is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)
    tf.logging.info("total_loss: %s", total_loss)
    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      if use_gpu and int(num_gpu_cores) >= 2:
          train_op = custom_optimization.create_optimizer(
              total_loss, learning_rate, num_train_steps, num_warmup_steps, fp16=fp16)
          output_spec = tf.estimator.EstimatorSpec(
              mode=mode,
              loss=total_loss,
              train_op=train_op)

      else:
          train_op = optimization.create_optimizer(
              total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

          output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              loss=total_loss,
              train_op=train_op,
              scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      if use_gpu and int(num_gpu_cores) >= 2:
          output_spec = tf.estimator.EstimatorSpec(
              mode=mode,
              loss=total_loss,
              eval_metrics_ops=eval_metrics[0](*eval_metrics[1]))
      else:
          output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              loss=total_loss,
              eval_metrics=eval_metrics,
              scaffold_fn=scaffold_fn)
    else:
      if use_gpu and int(num_gpu_cores) >= 2:
          output_spec = tf.estimator.EstimatorSpec(
              mode=mode,
              predictions={"probabilities": probabilities})
      else:
          output_spec = tf.contrib.tpu.TPUEstimatorSpec(
              mode=mode,
              predictions={"probabilities": probabilities},
              scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn




# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def loadJsonConfig(json_path):
    with open(json_path, 'r', encoding='utf-8') as file:
        config = json.load(file)
        return config


def generate_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return os.path.join(path, 'data')

def parse_path(path):
    if os.path.isdir(path):
        return os.path.join(path, 'data')
    else:
        return path

