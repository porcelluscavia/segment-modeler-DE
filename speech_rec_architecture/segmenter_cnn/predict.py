import tensorflow as tf
from processing import spectro_batch_generator
from config import Config
from model_architecture import Model

num_classes = 10


def predict(datadir):
    """
    Predicts labels, logits, and probabilties of spectrograms in folder.
    :param datadir:
    :return:
    """
    model = Model(Config, num_classes)

    X, Y = spectro_batch_generator(datadir,
                                   shuffle_files=False)  # TODO change from getting all files in a dir to passing wav files.

    estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=Config.logdir,
                                       config=tf.estimator.RunConfig().replace(save_summary_steps=10))

    if len(X.shape) != 3:
        X = tf.expand_dims(X, 0)

    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'images': X}, shuffle=False)

    return estimator.predict(input_fn)


if __name__ == '__main__':
    testdir = 'data/test_folder_single'

    prediction = predict(testdir)

    pred = next(prediction)

    # Should be '9'
    print('PREDICTED: %s | GS: %s' % (str(pred['class_ids'][0]), str(9)))

    testdir = 'data/test_folder_multi'

    prediction = predict(testdir)

    while True: # TODO Change this. Doesnt work.
        pred = next(prediction)
        print('PREDICTED: %s' % str(pred['class_ids'][0]))
