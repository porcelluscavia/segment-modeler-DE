import tensorflow as tf
from segment_modeler_DNN.segmenter_rnn.kiel_corpus_utils import features_and_labels_from_dir_kiel
from segment_modeler_DNN.segmenter_rnn.config import Config
from segment_modeler_DNN.segmenter_rnn.model import Model
from segment_modeler_DNN.segmenter_rnn.sampa_utils import SampaMapping
# # If using the console, you may need to append the path to the code.
import sys

sys.path.append("./segment_modeler_DNN")

tf.logging.set_verbosity(tf.logging.INFO)

# if __name__ == '__main__':
"""
Training method for model. All configurations can be edited in config_CNN.py
"""
# TODO Work on the data pipeline.
# TODO Timeseries? (also shuffling is not wanted if that is the case.
# TODO What to do if there is a model saved and I dont want to load it? Make folder name with model info?

####### LOAD AND PROCESS DATA #########

trainX, testX, trainY, testY = features_and_labels_from_dir_kiel(Config.datadir)

num_classes = len(SampaMapping.sampa_map.keys()) + 1
print("NUM CLASSES:", num_classes)

######## BUILD MODEL ############
# Declare model and pass in number of classes for model architecture.
model = Model(Config, num_classes)

estimator = tf.estimator.Estimator(model_fn=model.model_fn,
                                   model_dir=Config.logdir,
                                   config=tf.estimator.RunConfig().replace(save_summary_steps=50))

input_fn_train = tf.estimator.inputs.numpy_input_fn(
    x={'features': trainX},
    y=trainY,
    batch_size=Config.batch_size,
    num_epochs=None,
    shuffle=Config.shuffle)

input_fn_test = tf.estimator.inputs.numpy_input_fn(
    x={'features': testX},
    y=testY,
    batch_size=Config.batch_size,
    shuffle=Config.shuffle)

####### TRAIN MODEL ########
# Train the Model
num_steps = int(trainX.shape[0] / Config.batch_size)
for epoch in range(Config.epochs):
    estimator.train(input_fn_train, steps=None)

    ####### EVALUATE ON TEST DATA ##########
    e = estimator.evaluate(input_fn_test)
    print("Testing Accuracy:", e['accuracy'], '| Testing Loss:', e['loss'])
