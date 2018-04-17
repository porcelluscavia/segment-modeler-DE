
#read in sound file
#read in textgrid segmentation tier for
#inspu=iration: http://homepage.univie.ac.at/christian.herbst/python/#wavDemo

import praatTextGrid
from pydub import AudioSegment
import os
import sys
import numpy as np
from scipy.io import wavfile
#import praatUtil
import scipy.io.wavfile as wav


X_SIZE = 16000
IMG_SIZE = 64


AudioSegment.ffmpeg = "/opt/local/var/macports/sources/rsync.macports.org/macports/release/tarballs/ports/multimedia/ffmpeg"


def process_textgrid_and_wav(textgrids_dir, wavs_dir, test_wavs_storage_dir, train_wavs_storage_dir):
    """
     Extracts labels from textgrid, extracts timestamps for each labeled phoneme, calls method to create create sound file for each segment in larger file

     :returns list of train labels, list of test labels
     """


    try:
        all_train_labels = []
        all_test_labels = []
        for f_name in os.listdir(textgrids_dir):
            print("hello")
            #silly mac makes hidden configuration files that need to be ignored
            if not f_name.startswith('.'):
                print(textgrids_dir + f_name)

                all_times_of_test_clips = []
                all_times_of_train_clips = []
                labels_for_train = []
                labels_for_test = []
                # instantiate a new TextGrid object
                textGrid = praatTextGrid.PraatTextGrid(0, 0)

                arrTiers = textGrid.readFromFile(textgrids_dir + f_name)

                numTiers = len(arrTiers)
                print(numTiers)
                if numTiers != 2:
                    raise Exception("we expect two tiers in this file")

                    #use segments tier, the second tier in our textgrid file
                tier = arrTiers[1]

                for i in range(tier.getSize()):

                    #interval is list of start time, end time, segment annotation, in that order
                    interval = tier.get(i)
                    if tier.getSize() <= 1:
                            #ADD this later
                        interval[2] = "NONE"
                            #get_sound_clips(wav_path,interval[0],interval[1])
                    label = interval[2]
                    test = False
                    if label.startswith('ASF'):
                        test = True
                        test_clip_start_and_end = []
                        labels_for_test.append(label)
                        test_clip_start_and_end.append(interval[0])
                        test_clip_start_and_end.append(interval[1])


                    else:
                        train_clip_start_and_end = []
                        labels_for_train.append(label)
                        train_clip_start_and_end.append(interval[0])
                        train_clip_start_and_end.append(interval[1])

                    if test:
                        all_times_of_test_clips.append(test_clip_start_and_end)
                    else:
                        all_times_of_train_clips.append(train_clip_start_and_end)


                # print(len(labels_for_test))
                # print(len(all_times_of_test_clips))
                # print(len(labels_for_train))
                # print(len(all_times_of_train_clips))
                all_test_labels.append(labels_for_test)
                all_train_labels.append(labels_for_train)



                file_name_without_extension = os.path.splitext(os.path.basename(f_name))[0]
                #check if filtering through code also creates this _band extension
                wav_path = wavs_dir + file_name_without_extension + "_band.wav"
                print(wav_path)
                get_sound_clips(wav_path, all_times_of_test_clips, test_wavs_storage_dir)
                if all_times_of_train_clips:
                    get_sound_clips(wav_path, all_times_of_train_clips, train_wavs_storage_dir)


    except OSError:
    # If directory has already been created or is inaccessible
        if not os.path.exists(textgrids_dir):
            sys.exit("Error opening given textgrid file path")

    return all_train_labels, all_test_labels




def get_sound_clips(wav_path, clip_times, wavs_storage_dir, already_filtered=True):
    """
    Breaks existing sound files into many small wave files, one for each segment
    :returns nothing
    """

    try:
        song = AudioSegment.from_wav(wav_path)
        wav_name_without_extension = os.path.splitext(os.path.basename(wav_path))[0]

        # if not already_filtered:
        #     praatUtil.applyBandPassFilter(wav_path, 50,20000,20)


        for clip_time in clip_times:

            start = clip_time[0]
            end = clip_time[1]

            duration = end - start


            #reject clips equal to or over 2 seconds long
            if duration < 2:

                start_time_in_ms = start * 1000
                end_time_in_ms = end * 1000

                phoneme_segment = song[start_time_in_ms:end_time_in_ms]

                wav_name = wav_name_without_extension + str(start) + ".wav"
                phoneme_segment.export(wavs_storage_dir + wav_name, format="wav")



    except OSError:
        # If directory has already been created or is inaccessible
        if not os.path.exists(wav_path):
            sys.exit("Error opening wave file given path. Check whether all necessary files are in that directory")
        if not os.path.exists(wavs_storage_dir):
            sys.exit("The wav file storage directory does not exist in your file system. The wave files will not be saved.")

    return



#source: https://github.com/microic/niy/tree/master/examples/speech_commands_spectrogram

def spectrogram(wav_dir):
    list_of_specs = []
    for wav_file in os.listdir(wav_dir):
        framerate, wav_data = wavfile.read(wav_dir + wav_file)

        window_length = 512
        window_shift = 121

        if len(wav_data) > X_SIZE:
            wav_data = wav_data[:X_SIZE]

        X = np.zeros(X_SIZE).astype('float32')
        X[:len(wav_data)] += wav_data
        spec = np.zeros((IMG_SIZE, IMG_SIZE)).astype('float32')

        for i in range(IMG_SIZE):
            start = i * window_shift
            end = start + window_length
            sig = np.abs(np.fft.rfft(X[start:end] * np.hanning(window_length)))
            spec[:,i] = (sig[1:IMG_SIZE + 1])[::-1]

        spec = (spec-spec.min())/(spec.max()-spec.min())
        spec = np.log10((spec * 100 + 0.01))
        spec = (spec-spec.min())/(spec.max()-spec.min())
        list_of_specs.append(spec)

    return list_of_specs



def mfcc_batch_maker(wavs_storage_dir, labels):


    #number the labels
    #padded/zeroed np arrays
    #leave one-hotting to ryan
    # mfcc's of each segment
    #librosa or the other one?
    #make sure everything matches up!


    return


if '__main__' == __name__:
    #  MAKE SURE TO ADD DEFAULT FILTERED

    try:
        textgrids_dir, wavs_dir, train_wavs_storage_dir, test_wavs_storage_dir = sys.argv[1:]
    except ValueError:
        sys.exit("{} textgrids_dir, wavs_dir, train_wavs_storage_dir, test_wavs_storage_dir; Make sure the slashes are in this format (slash on end as well): '/Users/samski/Documents/Wavs_for_Model/'".format(sys.argv[0]))


    '''
    The way it's set up is that you have one directory with textgrids and one directory with wave files of the exact same name that have been filtered beforehand in Praat, 
    and thus have '_band' appended to the base name 
    Make sure the slashes are in this format (slash on end as well): '/Users/samski/Documents/Wavs_for_Model/'
    '''
    # #remove hardcoded paths later
    # textgrids_dir = '/Users/samski/Documents/Textgrids_for_Model/'
    # wavs_dir = '/Users/samski/Documents/Wavs_for_Model/'
    # #add error handing os.exist for this! must exist!
    # train_wavs_storage_dir = "/Users/samski/Documents/Train_Wavs/"
    # test_wavs_storage_dir = "/Users/samski/Documents/Test_Wavs/"


    labels = process_textgrid_and_wav(textgrids_dir, wavs_dir, train_wavs_storage_dir, test_wavs_storage_dir)

    train_labels = labels[0]
    test_labels = labels[1]

    # print(train_labels)
    # print(test_labels)

    # train_mfccs = mfcc_batch_maker(train_wavs_storage_dir, train_labels)
    # test_mfccs = mfcc_batch_maker(test_wavs_storage_dir, test_labels)

    all_specs = spectrogram(test_wavs_storage_dir)

    # print(all_specs)


