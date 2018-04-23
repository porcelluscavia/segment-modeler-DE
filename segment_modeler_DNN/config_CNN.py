class Config:
    """
    Configuration parameters
    """
    # Training Parameters
    learning_rate = 0.001
    num_steps = 2000
    dropout = 0.25
    epochs = 1000
    test_size = 0.33
    # Where the model and info is logged
    logdir = './models'
    # Where the training data is located
    datadir = 'C:/Users/ryanc/Dropbox/segment-modeler-DE/segment_modeler_DNN/data/train_wavs'
    img_size = (64, 64)
    x_size = 16000 # TODO Figure this out.
    shuffle = True

    # Network Parameters
    batch_size = 128
    # num_classes = 10
    # IMPORTANT: filter_size and kernals must be in arrays of the same length.
    filter_size = [32, 64]
    kernels = [5, 3]
    gru_include = False
    gru_size = 128


