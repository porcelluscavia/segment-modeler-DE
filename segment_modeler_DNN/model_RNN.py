import tensorflow as tf
from tensorflow.contrib import rnn
from segment_modeler_DNN.config_RNN import Config


class Model:

    def __init__(self,
                 config,
                 num_classes):
        self.config = config
        self.num_classes = num_classes

    def model_fn(self, features, labels, mode):
        """
        Builds the neural net
        :param features:
        :param labels:
        :param mode:
        :return:
        """
        print('labels', labels.shape)
        is_training = False

        if mode == tf.estimator.ModeKeys.TRAIN:
            is_training = True

        logits = rnn_net(features, self.num_classes, reuse=False, is_training=is_training)
        print('logits', logits.shape)
        # Predictions
        pred_classes = tf.argmax(logits, axis=1)
        pred_probs = tf.nn.softmax(logits)
        print('pred probs', pred_probs.shape)

        predictions = {
            'class_ids': pred_classes[:, tf.newaxis],
            'probabilities': pred_probs,
            'logits': logits,
        }

        # If prediction mode, early return
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Define loss and optimizer
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.cast(labels, dtype=tf.int32)))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        train_op = optimizer.minimize(loss,
                                      global_step=tf.train.get_global_step())

        # Evaluate the accuracy of the model
        accuracy = tf.metrics.accuracy(labels=labels, predictions=pred_classes, name='acc_op')

        metrics = {'accuracy': accuracy}
        tf.summary.scalar('accuracy', accuracy[1])
        tf.summary.scalar('loss', loss)

        merge_summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(self.config.logdir)

        summary_hook = tf.train.SummarySaverHook(save_steps=10, output_dir=self.config.logdir, summary_op='accuracy')

        logging_hook = tf.train.LoggingTensorHook({'accuracy': accuracy[0], 'loss': loss}, every_n_iter=10)

        # TF Estimators requires to return a EstimatorSpec, that specify
        # the different ops for training, evaluating, ...
        estim_specs = tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=pred_classes,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics,
            training_hooks=[logging_hook, summary_hook])

        return estim_specs


def gru_layer(size, activation=tf.nn.relu):
    """
    Creates a GRU layer.
    :param activation:
    :param size:
    :param input:
    :return:
    """
    return rnn.LSTMCell(size, activation=activation)


def gru_hidden_layers(inputs):
    print('gru lazyers', inputs.shape)
    cells = [gru_layer(size) for size in Config.hidden_sizes]
    print("cells", cells)
    cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
    print("cell", cell)
    hidden_layers, outputs = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    print('hidden out', hidden_layers)
    print('gru out', outputs)
    return hidden_layers, outputs


def rnn_net(features, n_classes, reuse, is_training):
    """
    Creates the layers for the RNN
    TODO Test out the GRU cell with time series data to see if it makes a difference.
    :param x_dict:
    :param n_classes:
    :param dropout:
    :param reuse:
    :param is_training:
    :param config:
    :return:
    """
    # Define a scope for reusing the variables
    with tf.variable_scope('GRUNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs

        print('features shape', features['features'].shape)
        features['features'] = tf.reshape(features['features'], shape=[-1, int(features['features'].shape[1]), 1])
        print('features shape2', features['features'].shape)

        hidden_layers, outputs = gru_hidden_layers(features['features'])
        print('len out', len(outputs))
        # Apply Dropout (if is_training is False, dropout is not applied)
        output = tf.layers.dropout(outputs[-1][-1], rate=Config.dropout, training=is_training)
        # Output layer, class prediction
        print('n_classes', n_classes)
        output = tf.layers.dense(output, n_classes, activation=None)
        print('output', output.shape)
        return output

