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
        if mode == tf.estimator.ModeKeys.TRAIN:
            self.is_training = True

        features = features['features']
        logits = self.conv_net(features, reuse=False, is_training=self.is_training)

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

        # TRAINING FUNCTIONS

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

    def conv_layer(self, filters, kernel, input, count):
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

    def conv_net(self, features, reuse, is_training):
        """
        Creates the layers for the CNN
        TODO Test out the GRU cell with time series data to see if it makes a difference.
        :param features:
        :param n_classes:
        :param reuse:
        :param is_training:
        :return:
        """
        # Define a scope for reusing the variables
        with tf.variable_scope('ConvNet', reuse=reuse):
            # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
            conv_layers = tf.reshape(features, shape=[-1, int(features.shape[1]), int(features.shape[2]), 1])

            # This is of variable size according to the Configs
            for i, f in enumerate(self.config.filter_size):
                conv_layers = self.conv_layer(f, self.config.kernels[i], conv_layers, str(i + 1))

            # Flatten the data to a 1-D vector for the fully connected layer
            fc1 = tf.contrib.layers.flatten(conv_layers)

            # Fully connected layer (in tf contrib folder for now)
            # TODO  should I make the fc layer variable?
            fc1 = tf.layers.dense(fc1, 1024)

            # Apply Dropout (if is_training is False, dropout is not applied)
            outputs = tf.layers.dropout(fc1, rate=self.config.dropout, training=is_training)

            # TODO Test this out. Might need to add size to Config. Uncomment to use
            if self.config.gru_include:
                gru_cell = rnn.GRUCell(64, activation=tf.nn.relu)
                _, outputs = tf.nn.dynamic_rnn(gru_cell, tf.expand_dims(fc1, axis=2), dtype=tf.float32)

            # Output layer, class prediction
            out = tf.layers.dense(outputs, self.num_classes, activation=None)

            return out
