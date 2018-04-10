class Config:
    """
    Configuration parameters
    """
    # Training Parameters
    learning_rate = 0.001
    num_steps = 2000
    dropout = 0.25
    epochs = 10
    test_size = 0.33
    # Where the model and info is logged
    logdir = './models'
    # Where the training data is located
    datadir = 'data/spoken_numbers_64x64'

    # Network Parameters
    batch_size = 128
    num_classes = 10
    # IMPORTANT: filter_size and kernals must be in arrays of the same length.
    filter_size = [16, 32, 64]
    kernels = [5, 5, 3]


