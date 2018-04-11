import tensorflow as tf
from sklearn.model_selection import train_test_split
from processing import spectro_batch_generator
from config import Config
from model_architecture import Model

tf.logging.set_verbosity(tf.logging.INFO)

if __name__ == '__main__':
    """
    Training method for model. All configurations can be edited in config.py
    """
    # TODO Work on the data pipeline.
    # TODO Timeseries? (also shuffling is not wanted if that is the case.
    # TODO What to do if there is a model saved and I dont want to load it? Make folder name with model info?

    ####### LOAD AND PROCESS DATA #########
    # Load data. Current data is 64x64 pixels TODO Sam and I need to work on this.
    # batch = mfcc_batch_generator('data/spoken_numbers_pcm', classes, batch_size=batch_size) # Maybe work with mfcc?
    # TODO Shuffling files? or time series?
    X, Y = spectro_batch_generator(Config.datadir,
                                   shuffle_files=True)  # TODO change from getting all files in a dir to passing wav files.
    # TODO choose image size and how to do cropping and padding?

    # Split data for testing and training.
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=Config.test_size, shuffle=True)

    # TODO Get Variable class size
    num_classes = 10

    ######## BUILD MODEL ############
    # Declare model and pass in number of classes for model architecture.
    model = Model(Config, num_classes)

    estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=Config.logdir,
                                       config=tf.estimator.RunConfig().replace(save_summary_steps=10))

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': trainX}, y=trainY,
        batch_size=Config.batch_size, num_epochs=Config.epochs, shuffle=False)

    ####### TRAIN MODEL ########
    # Train the Model
    estimator.train(input_fn, steps=Config.num_steps)

    ####### EVALUATE ON TEST DATA ##########

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': testX}, y=testY,
        batch_size=Config.batch_size, shuffle=False)
    # Use the Estimator 'evaluate' method
    e = estimator.evaluate(input_fn)

    print("Testing Accuracy:", e['accuracy'], '| Testing Loss:', e['loss'])
