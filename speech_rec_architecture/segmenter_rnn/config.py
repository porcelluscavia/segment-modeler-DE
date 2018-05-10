class Config:
    """
    Configuration parameters
    """
    architecture_type = 'rnn'

    # Training Parameters
    datadir = 'C:/Users/ryanc/Documents/kiel_corpus'  # Where the corpus is located
    # datadir = '/mnt/Shared/people/ryan/kiel_corpus'
    learning_rate = 0.001  # Learning rate for gradient descent
    dropout = 0.25  # Blanket dropout for all training layers.
    epochs = 50
    test_size = 0.2  # % of data set aside for training.
    shuffle = False  # If using time series (RNN), must be false.
    save_summary_steps = 50
    pred_type = ''

    # Network Parameters
    batch_size = 64
    hidden_sizes = [128, 128]  # Number of neurons per hidden layer. e.g. [256, 64] = 2 hidden layers
    bidirectional = False  # If you want a bidirectional GRU RNN. Takes much longer to train.
    cell_type = 'lstm'  # 'gru' 'rnn'
    intra_op_parallelism_threads = 1
    inter_op_parallelism_threads = 1
    device_count_cpu = 1

    # Data Params
    frame_size = 0.04  # 0.025 = 25 ms
    frame_overlap = 0.01  # 0.01 = 10 ms
    num_files = None  # If you want to limit the number of files used.
    feature_type = 'mfcc'  # 'spectrogram'
    mode_of_articulation = False  # Adds moa and poa to phonetic prediction model
