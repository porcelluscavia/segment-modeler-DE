import argparse
import sys
import numpy as np
from segmenter_cnn.train import train
from utils.kiel_corpus_utils import spectrograms_features_and_labels_from_dir_kiel
from utils.sampa_utils import SampaMapping
from segmenter_cnn.config import Config


def main(arg_parser):
    set_configs(arg_parser)

    if len(Config.kernels) != len(Config.filter_size):
        sys.exit('KERNAL SIZES AND FILTER SIZE NUMBER MUST MATCH')

    Config.pred_type = 'poa'
    features, labels = spectrograms_features_and_labels_from_dir_kiel(Config.datadir, Config.frame_size,
                                                                      Config.img_size, num_files=Config.num_files)
    Y = []
    for i, y in enumerate(labels):
        Y.append(SampaMapping.poa2idx[SampaMapping.sampa2poa[SampaMapping.idx2sampa[y]]])
    labels = np.array(Y)

    train(features, labels, Config)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def make_args(parser):
    # CONFIGS
    parser.add_argument('--data_dir', type=str, default=Config.datadir)
    parser.add_argument('--lr', type=float, default=Config.learning_rate)
    parser.add_argument('--do', type=float, default=Config.dropout)
    parser.add_argument('--ep', type=int, default=Config.epochs)
    parser.add_argument('--ts', type=float, default=Config.test_size)
    parser.add_argument('--shuffle', type=str2bool, default=Config.shuffle)
    parser.add_argument('--sum_steps', type=int, default=Config.save_summary_steps)
    parser.add_argument('--bs', type=int, default=Config.batch_size)
    parser.add_argument('--filter', type=int, nargs='+', default=Config.filter_size)
    parser.add_argument('--kernals', type=int, nargs='+', default=Config.kernels)
    parser.add_argument('--fs', type=float, default=Config.frame_size)
    parser.add_argument('--fo', type=float, default=Config.frame_overlap)
    parser.add_argument('--ft', type=str, default=Config.feature_type)
    parser.add_argument('--gru', type=str2bool, default=Config.gru_include)
    parser.add_argument('--gru_size', type=int, default=Config.gru_size)
    return parser


def set_configs(parser):
    Config.learning_rate = parser.parse_args().lr
    Config.dropout = parser.parse_args().do
    Config.epochs = parser.parse_args().ep
    Config.shuffle = parser.parse_args().shuffle
    Config.test_size = parser.parse_args().ts
    Config.save_summary_steps = parser.parse_args().sum_steps
    Config.batch_size = parser.parse_args().bs
    Config.filter_size = parser.parse_args().filter
    Config.kernels = parser.parse_args().kernals
    Config.frame_size = parser.parse_args().fs
    Config.frame_overlap = parser.parse_args().fo
    Config.feature_type = parser.parse_args().ft
    Config.gru_include = parser.parse_args().gru
    Config.gru_size = parser.parse_args().gru_size


if __name__ == '__main__':
    parser = make_args(argparse.ArgumentParser())

    set_configs(parser)

    main(parser)
