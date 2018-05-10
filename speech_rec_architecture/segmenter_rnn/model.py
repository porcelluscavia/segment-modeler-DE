import tensorflow as tf
from tensorflow.contrib import rnn


class Model:

    def __init__(self,
                 config,
                 num_classes,
                 logdir):
        """
        :param config: Configs from config.py
        :param num_classes:
        :param logdir:
        """
        self.config = config
        self.num_classes = num_classes
        self.logdir = logdir
        self.is_training = False

    def model_fn(self, features, labels, mode):
        """
        Builds the neural net
        :param features:
        :param labels:
        :param mode:
        :return:
        """
        if mode == tf.estimator.ModeKeys.TRAIN:  # If training, enables backprop, dropout, etc.
            self.is_training = True

        features = tf.reshape(features['features'], shape=[-1, int(features['features'].shape[1]), 1])
        logits = self.rnn_net(features, reuse=False, is_training=self.is_training)  # Loads model

        # Predictions
        pred_classes = tf.argmax(logits, axis=1)  # TODO should these be reversed? Do we use the logits to train????
        pred_probs = tf.nn.softmax(logits)

        predictions = {
            'class_ids': pred_classes[:, tf.newaxis],
            'probabilities': pred_probs,
            'logits': logits,
        }

        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:  # Returns predictions from model
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # TRAINING FUNCTIONS

        # Define loss and optimizer
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        train_op = optimizer.minimize(loss,
                                      global_step=tf.train.get_global_step())
        # Evaluate the accuracy of the model TODO improve this.
        accuracy = tf.metrics.accuracy(labels=labels, predictions=pred_classes, name='acc_op')
        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        tf.summary.scalar('loss', loss)

        tf.summary.merge_all()
        tf.summary.FileWriter(self.logdir)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics)
        return estim_specs

    def gru_hidden_layers(self, inputs, activation=tf.nn.relu):
        """
        Dynamically creates GRU layers, bidirectional and otherwise.
        :param inputs:
        :return:
        """
        # BIDIRECTIONAL GRU
        if self.config.cell_type == 'gru':
            rnn_cell = rnn.GRUCell
        elif self.config.cell_type == 'lstm':
            rnn_cell = rnn.LSTMCell
        else:
            rnn_cell = rnn.RNNCell

        if self.config.bidirectional:

            fcells = [rnn_cell(size, activation=activation) for size in self.config.hidden_sizes]  # Forward cells.
            if self.is_training:  # Dropout for training.
                fcells = [rnn.DropoutWrapper(cell, output_keep_prob=self.config.dropout,
                                             state_keep_prob=self.config.dropout) for cell in fcells]

            bcells = [rnn_cell(size, activation=activation) for size in self.config.hidden_sizes]  # Backward cells.
            if self.is_training:  # Dropout for training.
                bcells = [rnn.DropoutWrapper(cell, output_keep_prob=self.config.dropout,
                                             state_keep_prob=self.config.dropout) for cell in bcells]

            outputs, fstate, bstate = rnn.stack_bidirectional_dynamic_rnn(fcells, bcells, inputs, dtype=tf.float32,
                                                                          sequence_length=None)
            # TODO find out about sequence length
            return None, tf.concat(axis=2, values=outputs)
            # return None, tf.concat([fstate[-1], bstate[-1]], axis=1)  # Takes final state of birectional hidden layers

        # STANDARD GRU
        else:

            cells = [rnn_cell(size, activation=activation) for size in
                     self.config.hidden_sizes]  # Configures hidden cells.
            cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)  # Combines cells
            return tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

    def rnn_net(self, features, reuse, is_training):
        """
        Creates the layers for the RNN
        :param features:
        :param n_classes:
        :param reuse:
        :param is_training:
        :return:
        """
        # Define a scope for reusing the variables
        with tf.variable_scope('%sNet' % self.config.cell_type, reuse=reuse):

            # TF Estimator input is a dict, in case of multiple inputs
            hidden_layers, outputs = self.gru_hidden_layers(features['features'])

            if not self.config.bidirectional:  # Only need the final layer.
                outputs = outputs[-1]
                if self.config.cell_type == 'lstm' or self.config.cell_type == 'bnlstm':
                    outputs = outputs[-1]
            else:
                outputs = outputs[:, -1, :]

            # Apply Dropout (if is_training is False, dropout is not applied)
            output = tf.layers.dropout(outputs, rate=self.config.dropout, training=is_training)

            # Output layer, class prediction
            output = tf.layers.dense(output, self.num_classes, activation=None)
            return output
