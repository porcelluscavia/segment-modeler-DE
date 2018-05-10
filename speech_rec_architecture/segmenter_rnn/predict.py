import tensorflow as tf
from utils.sampa_utils import SampaMapping
from segmenter_rnn.config import Config
from segmenter_rnn.model import Model
from utils.mfcc_extraction import mfcc_features


def predict(features, model_dir):
    """
    Predicts labels, logits, and probabilties.
    :param features:
    :param model_dir:
    :return: An iterator with predictions
    """
    model = Model(Config, len(SampaMapping.sampa2idx.keys())+1, model_dir)
    estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir=model_dir,
                                       config=tf.estimator.RunConfig().replace(save_summary_steps=10))
    input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'features': features}, shuffle=False)
    return estimator.predict(input_fn)


def transform_data_mfcc(wavpath):
    """
    Fetches mfcc features from wav file.
    :param wavpath:
    :return: (mfcc_features, time_indices)
    """
    return mfcc_features(wavpath)

