import os
from random import shuffle
import skimage.io
import numpy as np


def dense_to_one_hot(labels_dense, num_classes=10):
    """
    Creates one-hot label vectors.
    :param labels_dense:
    :param num_classes:
    :return:
    """
    """Convert class labels from scalars to one-hot vectors."""
    return np.eye(num_classes)[labels_dense]


def spectro_batch_generator(data_dir, shuffle_files=False, batch_size=10, width=64):
    """
    Returns two arrays. One, a (data_size X hight_pixel X width_pixel) matrix of spectrogram images from directory
    along with their labels.
    TODO We need to streamline this pipeline so that we can take the spectrogram info directly from wav.
    :param data_dir:
    :param shuffle_files:
    :param batch_size:
    :param width:
    :return:
    """
    path = data_dir.replace("_spectros", "")  # HACK! remove!
    # height = width
    batch = []
    labels = []
    files = os.listdir(data_dir)
    # shuffle(files) # todo : split test_fraction batch here!
    print("Got %d source data files from %s" % (len(files), path))
    if shuffle_files:
        shuffle(files)
    for image_name in files:
        if not "_" in image_name: continue  # bad !?!
        image = skimage.io.imread(path + "/" + image_name).astype(np.float32)
        # image.resize(width,height) # lets see ...
        data = image / 255.  # 0-1 for Better convergence
        # data = data.reshape([width * height])  # tensorflow matmul needs flattened matrices wtf
        batch.append(list(data))
        # classe=(ord(image_name[0]) - 48)  # -> 0=0 .. A:65-48 ... 74 for 'z'
        label = (ord(image_name[0]) - 48) % 32  # -> 0=0  17 for A, 10 for z ;)
        labels.append(label)
    return np.array(batch), np.array(labels)

