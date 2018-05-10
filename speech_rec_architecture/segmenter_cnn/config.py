class Config:
    """
    Configuration parameters
    """
    architecture_type = 'cnn'

    # Training Parameters
    # datadir = 'C:/Users/ryanc/Documents/kiel_test'    # Where the training data is located
    datadir = '/mnt/Shared/people/ryan/kiel_corpus'
    learning_rate = 0.0001
    dropout = 0.2
    epochs = 50
    test_size = 0.2
    shuffle = True
    save_summary_steps = 50
    pred_type = ''

    # Network Parameters
    img_size = (12, 12)
    batch_size = 128
    #  IMPORTANT: filter_size and kernals must be in arrays of the same length.
    filter_size = [64, 32]
    kernels = [5, 3]
    intra_op_parallelism_threads = 1
    inter_op_parallelism_threads = 1
    device_count_cpu = 1
    gru_include = False
    gru_size = 128

    # Data Params

    frame_size = 0.025
    frame_overlap = 15
    num_files = None
    feature_type = 'spectrogram'
