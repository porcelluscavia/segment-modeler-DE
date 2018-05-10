import argparse
import numpy as np
from segmenter_rnn.train import train
from utils.kiel_corpus_utils import mfcc_features_and_labels_from_dir_kiel
from utils.sampa_utils import SampaMapping
from segmenter_rnn.config import Config


"""
Trains a model for predicting Place of Articulation
"""

def main(arg_parser):
    set_configs(arg_parser)
    Config.pred_type = 'poa'
    features, labels = mfcc_features_and_labels_from_dir_kiel(Config.datadir, Config.frame_size, num_files=Config.num_files)
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
    parser.add_argument('--sum_steps', type=int, default=Config.save_summary_steps)
    parser.add_argument('--bs', type=int, default=Config.batch_size)
    parser.add_argument('--hl', type=int, nargs='+', default=Config.hidden_sizes)
    parser.add_argument('--bidir', type=str2bool, default=Config.bidirectional)
    parser.add_argument('--cell', type=str, default=Config.cell_type)
    parser.add_argument('--fs', type=float, default=Config.frame_size)
    parser.add_argument('--fo', type=float, default=Config.frame_overlap)
    parser.add_argument('--ft', type=str, default=Config.feature_type)
    parser.add_argument('--moa', type=str, default=Config.mode_of_articulation)
    return parser


def set_configs(parser):
    Config.learning_rate = parser.parse_args().lr
    Config.dropout = parser.parse_args().do
    Config.epochs = parser.parse_args().ep
    Config.test_size = parser.parse_args().ts
    Config.save_summary_steps = parser.parse_args().sum_steps
    Config.batch_size = parser.parse_args().bs
    Config.hidden_sizes = parser.parse_args().hl
    Config.bidirectional = parser.parse_args().bidir
    Config.cell_type = parser.parse_args().cell
    Config.frame_size = parser.parse_args().fs
    Config.frame_overlap = parser.parse_args().fo
    Config.feature_type = parser.parse_args().ft
    Config.mode_of_articulation = parser.parse_args().moa


if __name__ == '__main__':
    parser = make_args(argparse.ArgumentParser())

    set_configs(parser)

    main(parser)
