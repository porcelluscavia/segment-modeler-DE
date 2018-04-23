from random import shuffle
import skimage.io
from natsort import natsorted
import praatTextGrid
from pydub import AudioSegment
import os
import sys
import numpy as np
import scipy.signal
from segment_modeler_DNN.sampa_include import SampaInclude
from segment_modeler_DNN.config_RNN import Config

# AudioSegment.ffmpeg = "/opt/local/var/macports/sources/rsync.macports.org/macports/release/tarballs/ports/multimedia/ffmpeg"
#
#
# def clean_data(spectrograms, labels):
#     """
#     Very specific for data
#     :param spectrograms:
#     :param labels:
#     :return:
#     """
#     spectrograms_new = []
#     labels_new = []
#     for i, l in enumerate(labels):
#         if l in SampaInclude.include:
#             spectrograms_new.append(spectrograms[i])
#             labels_new.append(l)
#     return np.array(spectrograms_new), np.array(labels_new)


# def spectrograms_and_labels_from_split(data_dir):
#     if not os.path.isdir(data_dir):
#         sys.exit('MUST BE DIRECTORY')
#     if 'labels.txt' not in os.listdir(data_dir):
#         sys.exit('DIRECTORY MUST INCLUDE labels.txt')
#     with open(os.path.join(data_dir, 'labels.txt'), 'r', encoding='utf-8') as l:
#         labels = [line.split('\t')[1].strip() for line in l.readlines()]
#     wav_files = [file for file in os.listdir(data_dir) if file.endswith('.wav')]
#     wav_files = sorted(wav_files, key=lambda x: int(x.split('_')[0]))
#     specs = [spectrogram(np.array(AudioSegment.from_wav(os.path.join(data_dir, f)).get_array_of_samples()))
#              for f in wav_files]
#     return specs, labels
#
#
# def spectrograms_and_labels_from_original(textgrid_path, wav_path, save_wavs=False, save_dir='.'):
#     """
#     This is specially designed for our data. TODO Make more flexible.
#     :param textgrid_path:
#     :param wav_path:
#     :param save_wavs:
#     :param save_dir:
#     :return:
#     """
#     tg_is_dir = os.path.isdir(textgrid_path)
#     wav_is_dir = os.path.isdir(wav_path)
#
#     if (tg_is_dir and wav_is_dir is False) or (tg_is_dir is False and wav_is_dir):
#         sys.exit('TEXTGRID AND WAV PATHS MUCH EITHER BOTH BE DIRECTORIES OR BOTH BE FILES.')
#     if save_wavs:
#         file = open(os.path.join(save_dir, ('labels.txt')), 'w', encoding='utf-8')
#         file.close()
#     if tg_is_dir:
#         specs = []
#         labels = []
#         durations = []
#         for tg in os.listdir(textgrid_path):
#             # silly mac makes hidden configuration files that need to be ignored
#             if tg.endswith('.TextGrid'):
#                 file_name_without_extension = os.path.splitext(os.path.basename(tg))[0]
#                 f_wav = os.path.join(wav_path, (file_name_without_extension + "_band.wav"))
#                 f_tg = os.path.join(textgrid_path, tg)
#                 s, l, d = process_textgrid_and_wav(f_tg, f_wav, save_wavs=save_wavs, save_dir=save_dir,
#                                                    total=len(specs))
#                 specs.extend(s)
#                 labels.extend(l)
#                 durations.extend(d)
#         return np.array(specs), np.array(labels), np.array(durations)
#     else:
#         return process_textgrid_and_wav(textgrid_path, wav_path, save_wavs=save_wavs, save_dir=save_dir)
#
#
# def process_textgrid_and_wav(textgrid_path, wav_path, save_wavs=False, save_dir='.', total=0):
#     """
#      Extracts labels from textgrid, extracts timestamps for each labeled phoneme, calls method to create create sound file for each segment in larger file
#
#      :returns list of train labels, list of test labels
#      """
#
#     # try:
#     labels = []
#     print('WORKING ON:', wav_path)
#     time_segments = []
#     durations = []
#
#     # instantiate a new TextGrid object
#     textGrid = praatTextGrid.PraatTextGrid(0, 0)
#     arrTiers = textGrid.readFromFile(textgrid_path)
#     numTiers = len(arrTiers)
#
#     if numTiers != 2:
#         raise Exception("we expect two tiers in this file")
#
#         # use segments tier, the second tier in our textgrid file
#     tier = arrTiers[1]
#
#     for i in range(tier.getSize()):
#         # interval is list of start time, end time, segment annotation, in that order
#         s_time, e_time, segment = tier.get(i)
#
#         labels.append(segment)  # Append label
#         time_segments.append({'start': s_time, 'end': e_time})
#         durations.append(e_time - s_time)
#
#     wavs = [w for w in
#             get_sound_clips(wav_path, time_segments, save=save_wavs, wavs_storage_dir=save_dir, labels=labels,
#                             total=total)]
#     spectrograms = [spectrogram(wav) for wav in wavs]
#     assert len(spectrograms) == len(labels)
#     return np.array(spectrograms), np.array(labels), np.array(durations)
#
#     #
#     # except OSError:
#     #     # If directory has already been created or is inaccessible
#     #     if not os.path.exists(textgrid_path):
#     #         sys.exit("Error opening given textgrid file path")
#
#
# def get_sound_clips(wav_path, clip_times, save=False, wavs_storage_dir='.', labels=None, total=0):
#     """
#     Breaks existing sound files into many small wave files, one for each segment
#     :returns nothing
#     """
#     if save and labels is None:
#         sys.exit('YOU MUST PROVIDE LABELS IN ORDER TO SAVE')
#
#     wav_full = AudioSegment.from_wav(wav_path)
#     wav_name_without_extension = os.path.splitext(os.path.basename(wav_path))[0]
#
#     for idx, clip_time in enumerate(clip_times):
#         if idx % 500 == 0:
#             print('WORKING ON %d of %d' % (idx, len(clip_times)))
#         start_time_in_ms = clip_time['start'] * 1000
#         end_time_in_ms = clip_time['end'] * 1000
#         phoneme_segment = wav_full[start_time_in_ms:end_time_in_ms]
#         if save:
#             with open(os.path.join(wavs_storage_dir, ('labels.txt')), 'a+', encoding='utf-8') as file:
#                 file.write('%d\t%s\n' % (total + idx, labels[idx]))
#                 file.close()
#             wav_save_path = '%d_%s_%s.wav' % (total + idx, wav_name_without_extension, str(clip_time['start']))
#             phoneme_segment.export(os.path.join(wavs_storage_dir, wav_save_path), format="wav")
#         yield np.array(phoneme_segment.get_array_of_samples())
#

def sample_spectrograms(wav_data):
    """
    Takes in a numpy array of audio frequency samples and returns a tuple of segment times,
    and a list of spectrograms.
    :param wav_data:
    :return: tuple (segment times, spectrograms)
    """
    _, t, s = scipy.signal.spectrogram(wav_data, nperseg=Config.sample_rate)
    return t, s.T

    # return scipy.signal.spectrogram(wav_data, nperseg=20)


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
    return labels


def features_and_labels_from_file_kiel(wav_path, s1h_path):
    wav_data = np.array(AudioSegment.from_wav(wav_path).get_array_of_samples())
    feature_times, features = sample_spectrograms(wav_data)
    sampa, sample_times = kiel_s1h_reader(s1h_path)
    labels = labels_from_samples(feature_times, sampa, sample_times)
    assert len(features) == len(labels)
    return features, labels


def features_and_labels_from_dir_kiel(kiel_dir):
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
    return np.array(features), labels


def spectrogram(wav_data):
    """
    source: https://github.com/microic/niy/tree/master/examples/speech_commands_spectrogram
    :param wav_dir:
    :return:
    """
    _, _, spec_raw = scipy.signal.spectrogram(wav_data)
    if len(spec_raw.shape) < 2:
        return np.zeros(Config.img_size).astype('float32')

    x_size, y_size = spec_raw.shape
    # y_size = spec_raw.shape[1]
    if x_size >= Config.img_size[0]:
        x_size = Config.img_size[0]
    if y_size >= Config.img_size[1]:
        y_size = Config.img_size[1]
    spec = np.zeros(Config.img_size).astype('float32')
    spec[:x_size, :y_size] = spec_raw[:x_size, :y_size]
    return spec

# def chunks(l, n):
#     """Yield successive n-sized chunks from l."""
#     for i in range(0, len(l), n):
#         yield np.array(l[i:i + n])

# def dense_to_one_hot(labels_dense, num_classes=10):
#     """
#     Creates one-hot label vectors.
#     :param labels_dense:
#     :param num_classes:
#     :return:
#     """
#     """Convert class labels from scalars to one-hot vectors."""
#     return np.eye(num_classes)[labels_dense]


# def spectro_batch_generator(data_dir, shuffle_files=False, batch_size=10, width=64):
#     """
#     Returns two arrays. One, a (data_size X hight_pixel X width_pixel) matrix of spectrogram images from directory
#     along with their labels.
#     TODO We need to streamline this pipeline so that we can take the spectrogram info directly from wav.
#     :param data_dir:
#     :param shuffle_files:
#     :param batch_size:
#     :param width:
#     :return:
#     """
#     path = data_dir.replace("_spectros", "")  # HACK! remove!
#     # height = width
#     batch = []
#     labels = []
#     files = os.listdir(data_dir)
#     # shuffle(files) # todo : split test_fraction batch here!
#     print("Got %d source data files from %s" % (len(files), path))
#     if shuffle_files:
#         shuffle(files)
#     for image_name in files:
#         if not "_" in image_name: continue  # bad !?!
#         image = skimage.io.imread(path + "/" + image_name).astype(np.float32)
#         # image.resize(width,height) # lets see ...
#         data = image / 255.  # 0-1 for Better convergence
#         # data = data.reshape([width * height])  # tensorflow matmul needs flattened matrices wtf
#         batch.append(list(data))
#         # classe=(ord(image_name[0]) - 48)  # -> 0=0 .. A:65-48 ... 74 for 'z'
#         label = (ord(image_name[0]) - 48) % 32  # -> 0=0  17 for A, 10 for z ;)
#         labels.append(label)
#     return np.array(batch), np.array(labels)
