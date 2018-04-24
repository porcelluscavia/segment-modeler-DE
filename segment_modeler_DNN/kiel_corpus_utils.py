from natsort import natsorted
from pydub import AudioSegment
import scipy
import os
import numpy as np
from segment_modeler_DNN.segmenter_rnn.config import Config
from segment_modeler_DNN.segmenter_rnn.sampa_utils import SampaMapping
from segment_modeler_DNN.segmenter_rnn.mfcc_extraction import mfcc_features


def kiel_s1h_reader(filename):
    """
    Takes in the filename of a Kiel corpus .S1H file and returns two lists of sampa and times
    :param filename:
    :return: list(sampa) list(sample_times)
    """
    sampa = []
    sample_times = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = iter(f.readlines())

    for l3 in lines:
        if l3.strip() == 'hend':
            _ = next(lines)
            break
    for l4 in lines:
        t, s = l4.strip().split()
        sample_times.append(t)
        if s[0] in ['#p:', '#c:, #l:']:
            s = '<p>'
        s = s.translate({ord(c): None for c in "#$+%-'"})
        s = SampaMapping.sampa_correction_map[s]
        sampa.append(s)  # Removes function stuff
    return sampa, sample_times


def labels_from_samples(feature_times, sampa, sample_times):
    """
    Gets a list of samples long with the sampa phoneme at each sample. <p> where nothing
    :param num_samples:
    :param sampa:
    :param sample_times:
    :return:
    """
    labels = []
    sampa_current = '<p>'
    idx = 0

    for ft in feature_times:
        if idx == len(sample_times) - 1:
            sampa_current = '<p>'
        elif ft >= int(sample_times[idx]):
            idx += 1
            sampa_current = sampa[idx]
        labels.append(sampa_current)
    return np.array([SampaMapping.sampa_map[s] for s in labels])


def features_and_labels_from_file_kiel(wav_path, s1h_path):
    """
    Returns mfcc features and labels from kiel corpus
    :param wav_path:
    :param s1h_path:
    :return:
    """
    # wav_data = np.array(AudioSegment.from_wav(wav_path).get_array_of_samples())
    # feature_times, features = sample_spectrograms(wav_data)
    features, feature_times = mfcc_features(wav_path, frame_size=Config.frame_size)
    sampa, sample_times = kiel_s1h_reader(s1h_path)
    labels = labels_from_samples(feature_times, sampa, sample_times)
    assert len(features) == len(labels)
    return features, labels


def sample_spectrograms(wav_data):
    """
    Takes in a numpy array of audio frequency samples and returns a tuple of segment times,
    and a list of spectrograms.
    :param wav_data:
    :return: tuple (segment times, spectrograms)
    """
    _, t, s = scipy.signal.spectrogram(wav_data, nperseg=Config.sample_rate)
    return t, s.T

def features_and_labels_from_dir_kiel(kiel_dir, train_test=True):
    """
    Returns features and labels from Kiel Corpus directory.
    Features are extracted from the spectrograms for every 20 samples.
    :param kiel_dir:
    :return:
    """
    features = []
    labels = []
    filenames = natsorted(set([f.split('.')[0] for f in os.listdir(kiel_dir)]))
    for i,fn in enumerate(filenames):
        if i % 500 == 0:
            print('PROCESSED %d of %d files' % (i, len(filenames)))
        wav_path = os.path.join(kiel_dir, '%s.wav' % fn)
        s1h_path = os.path.join(kiel_dir, '%s.S1H' % fn)
        if os.path.isfile(wav_path) and os.path.isfile(s1h_path):
            f, l = features_and_labels_from_file_kiel(wav_path, s1h_path)
            features.extend(f)
            labels.extend(l)
        else:
            print('%s does not have a wav and s1h file' % fn)
    assert len(features) == len(labels)
    if train_test:
        train_idx = int(len(features) * (1-0.8))
        trainX = np.array(features[:train_idx])
        trainY = np.array(labels[:train_idx])
        testX = np.array(features[train_idx:])
        testY = np.array(labels[train_idx:])
        return trainX, testX, trainY, testY
    return np.array(features), np.array(labels)