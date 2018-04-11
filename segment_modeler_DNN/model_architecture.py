import tensorflow as tf
from tensorflow.contrib import rnn


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

        is_training = False

        if mode == tf.estimator.ModeKeys.TRAIN:
            is_training = True

        logits = conv_net(features, self.num_classes, self.config.dropout, config=self.config, reuse=False,
                          is_training=is_training)

        # Predictions
        pred_classes = tf.argmax(logits, axis=1)
        pred_probs = tf.nn.softmax(logits)

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

        merge_summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(self.config.logdir)

        summary_hook = tf.train.SummarySaverHook(save_steps=10, output_dir=self.config.logdir, summary_op='accuracy')

        logging_hook = tf.train.LoggingTensorHook({'accuracy': accuracy[0]}, every_n_iter=10)

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


def conv_layer(filters, kernel, input, count):
    """
    Creates a convolutional layer.
    :param filters:
    :param kernel:
    :param input:
    :param count:
    :return:
    """
    convlayer = tf.layers.conv2d(input, filters, kernel, activation=tf.nn.relu, name='conv' + count)
    return tf.layers.max_pooling2d(convlayer, 2, 2, name='maxPool' + count)


def conv_net(x_dict, n_classes, dropout, reuse, is_training, config):
    """
    Creates the layers for the CNN
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
    with tf.variable_scope('ConvNet', reuse=reuse):
        # TF Estimator input is a dict, in case of multiple inputs
        X = x_dict['images']

        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        conv_layers = tf.reshape(X, shape=[-1, int(X.shape[1]), int(X.shape[2]), 1])

        # This is of variable size according to the Configs
        for i, f in enumerate(config.filter_size):
            conv_layers = conv_layer(f, config.kernels[i], conv_layers, str(i + 1))

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv_layers)

        # Fully connected layer (in tf contrib folder for now)
        # TODO  should I make the fc layer variable?
        fc1 = tf.layers.dense(fc1, 1024)

        # Apply Dropout (if is_training is False, dropout is not applied)
        outputs = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # TODO Test this out. Might need to add size to Config. Uncomment to use
        # gru_cell = rnn.GRUCell(64, activation=tf.nn.relu)
        # _, outputs = tf.nn.dynamic_rnn(gru_cell, tf.expand_dims(fc1, axis=2), dtype=tf.float32)

        # Output layer, class prediction
        out = tf.layers.dense(outputs, n_classes, activation=None)

    return out
