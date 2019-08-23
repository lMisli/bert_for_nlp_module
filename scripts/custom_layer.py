import tensorflow as tf


class custom_layer(object):
    """Layers for custom nlp tasks."""
    def __init__(self, input=None, label=None, loss_func=None, is_training=None, num_units=None, num_layers=None, dropout_rate=0.9, lstm_out='avg'):
        """

        :param input:
        :param label:
        :param loss_func:
        :param is_training:
        :param num_units:
        :param num_layers:
        :param dropout_rate:
        :param lstm_out: str, using the avg/sum/concat value of time step out or final state as lstm output. 'avg'(default),'sum','concat' ,'final' can be choosed.
        """
        self.input = None
        if input != None:
            self.input = tf.cast(input,dtype=tf.float32)
        self.label = None
        if label != None:
            self.label = tf.cast(label,dtype=tf.float32)
        self.loss_func = loss_func
        self.is_training = is_training
        self.num_units = num_units
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.lstm_out = lstm_out

    def dense(self):
        with tf.variable_scope('dense_layer'):
            output_weights = tf.get_variable("output_weights", [self.label.shape[-1].value, self.input.shape[-1].value], initializer=tf.truncated_normal_initializer(stddev=0.02))
            output_bias = tf.get_variable("output_bias", [self.label.shape[-1].value], initializer=tf.zeros_initializer())
            if self.is_training:
                self.input = tf.nn.dropout(self.input, keep_prob=self.dropout_rate)
            logits = tf.matmul(self.input, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

            probabilities = tf.nn.sigmoid(logits, name="probability")

        return logits, probabilities

    def get_lstm_cell(self):
        cell = tf.nn.rnn_cell.LSTMCell(self.num_units)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_rate)
        return cell

    def lstm(self):

        with tf.variable_scope('lstm_layer'):

            cell = tf.nn.rnn_cell.MultiRNNCell([self.get_lstm_cell() for _ in range(self.num_layers)], state_is_tuple=True)
            self.input = tf.nn.dropout(self.input, keep_prob=self.dropout_rate)
            step_outputs, final_state = tf.nn.dynamic_rnn(cell, self.input, dtype=tf.float32)

            outputs = self.get_lstm_out(step_outputs, final_state)

            return outputs

    def bilstm(self):
        with tf.variable_scope('lstm_layer'):
            fw_cell = tf.nn.rnn_cell.MultiRNNCell([self.get_lstm_cell() for _ in range(self.num_layers)], state_is_tuple=True)
            bw_cell = tf.nn.rnn_cell.MultiRNNCell([self.get_lstm_cell() for _ in range(self.num_layers)], state_is_tuple=True)

            step_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, dtype=tf.float32)

            outputs = self.get_lstm_out(step_outputs, final_state)
            return outputs

    def get_lstm_out(self, step_outputs, final_state):
        if self.lstm_out == 'concat':
            outputs = tf.reshape(step_outputs, shape=[self.input.shape[0].value,-1])
        elif self.lstm_out == 'sum':
            outputs = tf.reduce_sum(step_outputs, axis=-2)
        elif self.lstm_out == 'final':
            outputs = final_state
        else:
            outputs = tf.reduce_mean(step_outputs, axis=-2)
        return outputs

    def get_loss_func(self):
        per_example_loss = None
        loss = None
        with tf.variable_scope('loss'):
            if self.is_training:
                per_example_loss = self.loss_func(labels=self.label, logits=self.input, name="loss")
                loss = tf.reduce_mean(per_example_loss)

        return loss, per_example_loss















