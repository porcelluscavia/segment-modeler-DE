
#read in sound file
#read in textgrid segmentation tier for
#inspu=iration: http://homepage.univie.ac.at/christian.herbst/python/#wavDemo

import praatTextGrid
from pydub import AudioSegment
import os
import sys
#import praatUtil
import scipy.io.wavfile as wav


AudioSegment.ffmpeg = "/opt/local/var/macports/sources/rsync.macports.org/macports/release/tarballs/ports/multimedia/ffmpeg"

#from Daniel de Kok's Deep Learning Course
class Numberer:
    def __init__(self):
        self.v2n = dict()
        self.n2v = list()
        self.start_idx = 1

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

def process_textgrid_and_wav(textgrid_path):
    #textgrid_path, wav_path
    # path that contains all the annotated wave files
    path = '/Users/samski/Documents/Textgrids_for_Model'

    #ADD ERROR HANDLING
    try:

        for f_name in os.listdir(path):

                all_times_of_clips = []
                # instantiate a new TextGrid object
                textGrid = praatTextGrid.PraatTextGrid(0, 0)

                arrTiers = textGrid.readFromFile('/Users/samski/Documents/Textgrids_for_Model/' + f_name)

                numTiers = len(arrTiers)
                print(numTiers)
                if numTiers != 2:
                    raise Exception("we expect two tiers in this file")

                #use segments tier
                tier = arrTiers[1]

                for i in range(tier.getSize()):
                    clip_start_and_end = []
                    #interval is list of start time, end time, segment annotation, in that order
                    interval = tier.get(i)
                    if tier.getSize() <= 1:
                        #ADD this later
                        interval[2] = "NONE"
                        #get_sound_clips(wav_path,interval[0],interval[1])
                        #print("\t", interval[2])


                    clip_start_and_end.append(interval[0])
                    clip_start_and_end.append(interval[1])

                    all_times_of_clips.append(clip_start_and_end)

                file_name_without_extension = os.path.splitext(os.path.basename(f_name))[0]
                wav_path = '/Users/samski/Documents/Wavs_for_Model/' + file_name_without_extension + "_band.wav"
                get_sound_clips(wav_path, all_times_of_clips)

    except OSError:
    # If directory has already been created or is inaccessible
        if not os.path.exists(path):
            sys.exit("Error opening given textgrid file path")

    return all_times_of_clips


def get_sound_clips(wav_path, clip_times, already_filtered=True):

    #filter out irrelevant frequencies!!!!!!!!!!!!!!!
    try:
        song = AudioSegment.from_wav(wav_path)
        wav_name_without_extension = os.path.splitext(os.path.basename(f_name))[0]

        #song = AudioSegment.from_wav(path)

        # if not already_filtered:
        #     praatUtil.applyBandPassFilter(wav_path, 50,20000,20)

        for clip_time in clip_times:

            start = clip_time[0]
            end = clip_time[1]

            start_time_in_ms = start * 1000
            end_time_in_ms = end * 1000

            phoneme_segment = song[start_time_in_ms:end_time_in_ms]

        # (rate, sig) = wav.read("file.wav")
        # mfcc_feat = mfcc(sig, rate)
        # fbank_feat = logfbank(sig, rate)


            wav_name = wav_name_without_extension + start + ".wav"
            phoneme_segment.export("/Users/samski/Documents/Wavs_for_Model/" + wav_name, format="wav")

    except OSError:
        # If directory has already been created or is inaccessible
        if not os.path.exists(path):
            sys.exit("Error opening wave file given path")
    return


def get_mfccs():
    return


if '__main__' == __name__:
    #  MAKE SURE TO ADD DEFAULT FILTERED
    #remove hardcoded paths later
    textgrid_path = '/Users/samski/Documents/Textgrids_for_Model'
    #wav_path = "/Users/samski/Documents/Wavs_for_Model/rec_006_IS_id_011_1.wav"

    clip_times = process_textgrid_and_wav(textgrid_path)
    # clip_times = [[10, 15]]

    #
    #get_sound_clips(wav_path, clip_times)


