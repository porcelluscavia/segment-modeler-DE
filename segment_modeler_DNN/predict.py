import tensorflow as tf
from segment_modeler_DNN.segmenter_rnn.sampa_utils import SampaMapping
from segment_modeler_DNN.segmenter_rnn.config import Config
from segment_modeler_DNN.segmenter_rnn.model import Model
from segment_modeler_DNN.segmenter_rnn.mfcc_extraction import mfcc_features


def predict(features):
    """
    Predicts labels, logits, and probabilties.
    :param datadir:
    :return:
    """
    model = Model(Config, len(SampaMapping.sampa_map.keys())+1)
    estimator = tf.estimator.Estimator(model_fn=model.model_fn, model_dir='C:/Users/ryanc/Dropbox/segment-modeler-DE/segment_modeler_DNN/models',
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


if __name__ == '__main__':
    wavpath = 'C:/Users/ryanc/Dropbox/segment-modeler-DE/kiel_corpus/DLME088.wav'

    features, times = transform_data_mfcc(wavpath)
    prediction = predict(features)

    from itertools import groupby
    preds1 = [i['class_ids'] for i in prediction]
    preds2 = [i[0][0] for i in groupby(preds1)]
    preds3 = [SampaMapping.sampa_map_from_idx[i] for i in preds2]
    pred4 = [i for i in preds3 if i != '<p>']
    print(''.join(pred4))