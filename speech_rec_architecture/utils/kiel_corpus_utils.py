from natsort import natsorted
import scipy
import os
import numpy as np
import scipy.signal
from utils.sampa_utils import SampaMapping
from utils.mfcc_extraction import mfcc_features
from utils.spectrogram_utils import spectrogram, pad_matrix
from utils.audio_utils import get_signal


def kiel_s1h_reader(filename):
    """
    Takes in the filename of a Kiel corpus .S1H file and returns two lists of sampa and times
    :param filename:
    :return: list(sampa) list(sample_times)
    """
    sampa = []  # Store sampa phonemes in list
    sample_times = []  # The times when the sampa phonemes start TODO Make sure it is actually when they start
    with open(filename, 'r', encoding='utf-8') as f:  # Open the .S1H file from the Kiel corps
        lines = iter(f.readlines())  # Reads lines into iterative stack

    for l3 in lines:
        if l3.strip() == 'hend':  # Removes stack until it reaches 'hend'. Other sections irrelevant
            _ = next(lines)
            break
    for l4 in lines:  # Stores sampa and times into lists.
        t, s = l4.strip().split()
        sample_times.append(t)
        if s[0] in ['#p:', '#c:, #l:']:  # TODO Find set of nonlinguistic cues.
            s = '<p>'
        s = s.translate({ord(c): None for c in "#$+%-'"})
        s = SampaMapping.sampa_correction[s]
        sampa.append(s)  # Removes function stuff
    return sampa, sample_times


def labels_from_samples(feature_times, sampa, sample_times):
    """
    Gets a list of samples long with the sampa phoneme at each sample. <p> where nothing
    :param feature_times:
    :param sampa:
    :param sample_times:
    :return:
    """
    labels = []
    sampa_current = '<p>'  # Start of .wavs are empty space. Disregard and maybe TODO Remove from training?
    idx = 0

    for ft in feature_times:
        if idx == len(sample_times) - 1:
            sampa_current = '<p>'
        elif ft >= int(sample_times[idx]):
            idx += 1
            sampa_current = sampa[idx]
        labels.append(sampa_current)
        # Maps to indices before returning all sampa labels.
    return np.array([SampaMapping.sampa2idx[s] for s in labels])


def mfcc_features_and_labels_from_file_kiel(wav_path, s1h_path, frame_size):
    """
    Returns mfcc features and labels from kiel corpus
    :param wav_path:
    :param s1h_path:
    :param frame_size:
    :return:
    """
    signal, sample_rate = get_signal(wav_path)
    features, feature_times = mfcc_features(signal, sample_rate, frame_size=frame_size)
    sampa, sample_times = kiel_s1h_reader(s1h_path)
    labels = labels_from_samples(feature_times, sampa, sample_times)
    assert len(features) == len(labels)  # Sanity check
    return features, labels


# def spectrograms_features_and_labels_from_file_kiel(wav_path, s1h_path, img_size, frame_size, overlap=15):
#
#     sampa, sample_times = kiel_s1h_reader(s1h_path)
#     sample_times = [int(i) for i in sample_times]
#     wav_data, sample_rate = get_signal(wav_path)
#     overlap = int((sample_rate/1000) * overlap)
#     sample_slices = [(sample_times[i]-overlap, sample_times[i+1]+overlap) for i in range(len(sample_times)) if i+1 < len(sample_times)]
#     features = []
#     for i, j in sample_slices:
#         spec, _, _ = spectrogram(wav_data[i:j], sample_rate, frame_size)
#         features.append(pad_matrix(spec, img_size))
#         # features.append(spec)
#     assert len(features) == len(sampa[:-1])  # Sanity check
#     return features, sampa[:-1], sample_times


def spectrograms_features_and_labels_from_file_kiel(wav_path, s1h_path, img_size, frame_size, overlap=15):

    sampa, sample_times = kiel_s1h_reader(s1h_path)
    sample_times = [int(i) for i in sample_times]
    wav_data, sample_rate = get_signal(wav_path)
    overlap = int((sample_rate/1000) * overlap)
    sample_slices = [(sample_times[i]-overlap, sample_times[i+1]+overlap) for i in range(len(sample_times)) if i+1 < len(sample_times)]
    features = []
    for i, j in sample_slices:
        f, _ = mfcc_features(wav_data[i:j], sample_rate, frame_size=frame_size)
        features.append(pad_matrix(f, img_size))
    #TODO padding
    # assert len(features) == len(sampa[:-1])  # Sanity check
    return features, sampa[:-1], sample_times


def spectrograms_features_and_labels_from_dir_kiel(kiel_dir, frame_size, img_size, num_files=None, overlap=15):
    """
    Returns features and labels from Kiel Corpus directory.
    Features are extracted from the spectrograms for every 20 samples.
    :param kiel_dir:
    :param frame_size:
    :type num_files: object
    :return:
    """
    features = []
    labels = []
    filenames = natsorted(set([f.split('.')[0] for f in os.listdir(kiel_dir)]))
    if num_files is not None:
        filenames = filenames[:num_files]
    for i, fn in enumerate(filenames):
        if i % 500 == 0:
            print('PROCESSED %d of %d files' % (i, len(filenames)))
        wav_path = os.path.join(kiel_dir, '%s.wav' % fn)
        s1h_path = os.path.join(kiel_dir, '%s.S1H' % fn)
        if os.path.isfile(wav_path) and os.path.isfile(s1h_path):
            f, l, _ = spectrograms_features_and_labels_from_file_kiel(wav_path, s1h_path, img_size,
                                                                      frame_size, overlap=overlap)
            features.extend(f)
            labels.extend(l)
        else:
            print('%s does not have a wav and s1h file' % fn)

    return np.array(features), np.array([SampaMapping.sampa2idx[l] for l in labels])


def mfcc_features_and_labels_from_dir_kiel(kiel_dir, frame_size, num_files=None):
    """
    Returns features and labels from Kiel Corpus directory.
    Features are extracted from the spectrograms for every 20 samples.
    :param kiel_dir:
    :param frame_size:
    :type num_files: object
    :return:
    """
    features = []
    labels = []
    filenames = natsorted(set([f.split('.')[0] for f in os.listdir(kiel_dir)]))
    if num_files is not None:
        filenames = filenames[:num_files]
    for i, fn in enumerate(filenames):
        if i % 500 == 0:
            print('PROCESSED %d of %d files' % (i, len(filenames)))
        wav_path = os.path.join(kiel_dir, '%s.wav' % fn)
        s1h_path = os.path.join(kiel_dir, '%s.S1H' % fn)
        if os.path.isfile(wav_path) and os.path.isfile(s1h_path):
            f, l = mfcc_features_and_labels_from_file_kiel(wav_path, s1h_path, frame_size)
            features.extend(f)
            labels.extend(l)
        else:
            print('%s does not have a wav and s1h file' % fn)

    return np.array(features), np.array(labels)


if __name__ == '__main__':

    dir_path = 'C:/Users/ryanc/Documents/kiel_corpus'
    feats, labs = spectrograms_features_and_labels_from_dir_kiel(dir_path, 0.025, (210, 150))
    sizes = [i.shape for i in feats]
    x = [i[0] for i in sizes]
    y = [i[1] for i in sizes if len(i) > 1]

    features = []
    for f in feats:
        features.append(pad_matrix(f, (12,12)))
