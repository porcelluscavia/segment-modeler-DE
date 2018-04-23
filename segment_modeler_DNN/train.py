# # If using the console, you may need to append the path to the code.
import sys

sys.path.append("./segment_modeler_DNN")

import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from segment_modeler_DNN.processing import features_and_labels_from_dir_kiel #, spectrograms_and_labels_from_split, clean_data
from segment_modeler_DNN.config_RNN import Config
from segment_modeler_DNN.model_RNN import Model
from segment_modeler_DNN.numberer import Numberer

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    """
    Training method for model. All configurations can be edited in config_CNN.py
    """
    # TODO Work on the data pipeline.
    # TODO Timeseries? (also shuffling is not wanted if that is the case.
    # TODO What to do if there is a model saved and I dont want to load it? Make folder name with model info?

    ####### LOAD AND PROCESS DATA #########
    # Load data. Current data is 64x64 pixels TODO Sam and I need to work on this.
    # batch = mfcc_batch_generator('data/spoken_numbers_pcm', classes, batch_size=batch_size) # Maybe work with mfcc?
    # TODO Shuffling files? or time series?
    # X, Y = spectro_batch_generator(Config.datadir,
    #                                shuffle_files=True)  # TODO change from getting all files in a dir to passing wav files.
    # TODO choose image size and how to do cropping and padding?

    X, Y = features_and_labels_from_dir_kiel(Config.datadir)

    # X, Y = clean_data(X, Y) # For CNN

    X[np.isnan(X)] = 0.0 # NaNs in data

    n = Numberer()

    Y = np.array([n.number(y, add_if_absent=True) for y in Y])

    num_classes = n.max_number()
    print("NUM CLASSES:", num_classes)

    # Split data for testing and training.
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=Config.test_size, shuffle=Config.shuffle)

    # TODO Get Variable class size

    ######## BUILD MODEL ############
    # Declare model and pass in number of classes for model architecture.
    model = Model(Config, num_classes)

    estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=Config.logdir,
                                       config=tf.estimator.RunConfig().replace(save_summary_steps=10))

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'features': trainX},
        y=trainY,
        batch_size=Config.batch_size,
        num_epochs=Config.epochs,
        shuffle=Config.shuffle)

    ####### TRAIN MODEL ########
    # Train the Model
    estimator.train(input_fn,
                    steps=Config.num_steps)

    ####### EVALUATE ON TEST DATA ##########

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'features': testX},
        y=testY,
        batch_size=Config.batch_size,
        shuffle=Config.shuffle)
    # Use the Estimator 'evaluate' method
    e = estimator.evaluate(input_fn)

    print("Testing Accuracy:", e['accuracy'], '| Testing Loss:', e['loss'])
