# read in sound file
# read in textgrid segmentation tier for
# inspiration: http://homepage.univie.ac.at/christian.herbst/python/#wavDemo

import praatTextGrid
from pydub import AudioSegment
import os
import sys

import numpy as np
from process_data import process_textgrid_and_wav
from mfcc_extraction import mfcc_features


from segment_modeler_DNN.sampa_utils import SampaMapping
import numpy as np
from scipy.io import wavfile
# import praatUtil
import scipy.io.wavfile as wav

# process_textgrid_and_wav("/Users/samski/Documents/Textgrids_fooling/","/Users/samski/Documents/Wavs_fooling/")


X_SIZE = 16000
IMG_SIZE = 64

AudioSegment.ffmpeg = "/opt/local/var/macports/sources/rsync.macports.org/macports/release/tarballs/ports/multimedia/ffmpeg"


class Numberer:
    def __init__(self):
        self.v2n = dict()
        self.n2v = list()
        self.start_idx = 71

    def number(self, value, add_if_absent=True):
        n = self.v2n.get(value)

        if n is None:
            if add_if_absent:
                n = len(self.n2v) + self.start_idx
                self.v2n[value] = n
                self.n2v.append(value)
            else:
                return 0

        return n

    def value(self, number):
        self.n2v[number - 1]

    def max_number(self):
        return len(self.n2v) + 1



def process_textgrid_and_wav(textgrids_dir, wavs_dir, test_wavs_storage_dir=None, train_wavs_storage_dir=None,
                             wavs_exist=True, need_numbers= False):
    """
     Extracts labels from textgrid, extracts timestamps for each labeled phoneme, calls method to create create sound file for each segment in larger file

     :returns list of train labels, list of test labels
     """

    try:
        all_labels = []


        for f_name in os.listdir(textgrids_dir):
            # silly mac makes hidden configuration files that need to be ignored
            if not f_name.startswith('.'):
                print(textgrids_dir + f_name)

                all_times_of_clips = []
                labels_num = Numberer()

                # instantiate a new TextGrid object
                textGrid = praatTextGrid.PraatTextGrid(0, 0)

                arrTiers = textGrid.readFromFile(textgrids_dir + f_name)

                numTiers = len(arrTiers)
                print(numTiers)
                if numTiers != 2:
                    raise Exception("we expect two tiers in this file")

                    # use segments tier, the second tier in our textgrid file
                tier = arrTiers[1]

                for i in range(tier.getSize()):

                    # interval is list of start time, end time, segment annotation, in that order
                    interval = tier.get(i)
                    if tier.getSize() <= 1:
                        # ADD this later
                        interval[2] = "NONE"
                        # get_sound_clips(wav_path,interval[0],interval[1])
                    label = interval[2]

                    all_labels.append(label)
                    all_times_of_clips.append(interval[0] * 10000)


                file_name_without_extension = os.path.splitext(os.path.basename(f_name))[0]
                # check if filtering through code also creates this _band extension
                wav_path = wavs_dir + file_name_without_extension + "_band.wav"
                # print(wav_path)

                if not wavs_exist:
                    get_sound_clips(wav_path, all_times_of_test_clips, test_wavs_storage_dir)
                    if all_times_of_train_clips:
                        get_sound_clips(wav_path, all_times_of_train_clips, train_wavs_storage_dir)

                if need_numbers:
                    all_labels = number_labels(all_labels, labels_num)



    except OSError:
        # If directory has already been created or is inaccessible
        if not os.path.exists(textgrids_dir):
            sys.exit("Error opening given textgrid file path")


    return (all_labels, all_times_of_clips)
    
    
    
def labels_from_samples_KEC(feature_times, kec_labels, sample_times):
    labels = []
    label_current = "0"
    idx = 0


    for ft in feature_times:
        if idx == len(sample_times) - 1:
            label_current = '0'
        elif ft >= int(sample_times[idx]):

            idx += 1
            # if label starts with asf

            label_current = kec_labels[idx]
            if not label_current.startswith('ASF'):
                # 1 means there was an actual segment there
                label_current = '1'

        labels.append(label_current)
    return np.array(labels)

def keep_only_asf_segs(feature_times, kec_labels, sample_times):
    labels = labels_from_samples_KEC(feature_times, kec_labels, kec_sample_times)

    return np.array(labels)


def features_and_labels_from_KEC(dir=None, need_numbers=False):
    kec_labels, kec_sample_times = process_textgrid_and_wav("/Users/samski/Documents/Textgrids_fooling/","/Users/samski/Documents/Wavs_fooling/", True)
    features, feature_times = mfcc_features("/Users/samski/Documents/Wavs_fooling/rec_004_AS_id_008_1_band.wav", frame_size= 0.025)
    labels = labels_from_samples_KEC(feature_times, kec_labels, kec_sample_times)
    assert len(features) == len(labels)
    return features, labels
