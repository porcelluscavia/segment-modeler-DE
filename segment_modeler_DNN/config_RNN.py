class Config:
    """
    Configuration parameters
    """
    # Training Parameters
    learning_rate = 0.001
    num_steps = 200
    dropout = 0.25
    epochs = 1000
    test_size = 0.2
    # Where the model and info is logged
    logdir = './models'
    # Where the training data is located
    datadir = 'C:/Users/ryanc/Dropbox/segment-modeler-DE/kiel_corpus'
    shuffle = False

    # Network Parameters
    batch_size = 512
    hidden_sizes = [256, 64, 128]

    # Data Params
    sample_rate = 20


