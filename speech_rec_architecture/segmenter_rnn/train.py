import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from segmenter_rnn.model import Model
from utils.network_utils import logdir

tf.logging.set_verbosity(tf.logging.INFO)


def train(features, labels, config):
    """
    Training method for model. All configurations can be edited in config.py
    """
    print('x shape', features.shape)
    num_classes = np.amax(labels) + 1  # Need to add 0 because indices start at 1 TODO FIX
    print('Num Classes:', num_classes)

    train_x, test_x, train_y, test_y = train_test_split(features, labels, test_size=config.test_size, shuffle=False)

    model_dir = logdir(config)  # Makes the logging directory with hyperparams

    # Declare model and pass in number of classes for model architecture.
    model = Model(config, num_classes, model_dir)

    # Adds model configs
    session_config = tf.ConfigProto(intra_op_parallelism_threads=config.intra_op_parallelism_threads,
                                    inter_op_parallelism_threads=config.inter_op_parallelism_threads,
                                    allow_soft_placement=True,
                                    device_count={'CPU': config.device_count_cpu}
                                    )

    estimator = tf.estimator.Estimator(model_fn=model.model_fn,
                                       model_dir=model_dir,
                                       config=tf.estimator.RunConfig().replace(
                                           save_summary_steps=config.save_summary_steps,
                                           session_config=session_config)
                                       )

    input_fn_train = tf.estimator.inputs.numpy_input_fn(
        x={'features': train_x},
        y=train_y,
        batch_size=config.batch_size,
        num_epochs=None,
        shuffle=config.shuffle)

    input_fn_test = tf.estimator.inputs.numpy_input_fn(
        x={'features': test_x},
        y=test_y,
        batch_size=config.batch_size,
        shuffle=config.shuffle)

    '''
    TRAIN MODEL
    '''
    # Train the Model
    num_steps = int(train_x.shape[0] / config.batch_size)  # Figure out how many steps in epoch
    print('NUM STEPS', num_steps)

    # Training loop
    for epoch in range(config.epochs):
        estimator.train(input_fn_train, steps=num_steps)  # Train for one epoch

        '''
        EVALUATE ON TEST DATA
        '''
        e = estimator.evaluate(input_fn_test)  # Eval after epoch

        print("Testing Accuracy:", e['accuracy'], '| Testing Loss:', e['loss'])
