from __future__ import print_function
from keras import backend as K
from keras.utils import np_utils
from trainer.load_data import load_dataset
from trainer.models import create_model, compile_model, evaluate_model
from keras.utils import multi_gpu_model
import tensorflow as tf

K.set_image_dim_ordering('th')


def dispatch():
    # Load our train/test data.
    (X_train, y_train), (X_test, y_test) = load_dataset(cloud_train)

    # One-hot encode the category outputs.
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # Create the model.
    with tf.device('/cpu:0'):
        model = create_model(10)

    # If gpu count >=2, return the multi_gpu_model. Keras uses regular model for gpu counts <=1.
    # 9+ gpus is not supported so Keras will throw an error prompting the user to enter valid gpu #.
    if gpus >= 2:
        model = multi_gpu_model(model, gpus)

    # Compile the model.
    model = compile_model(model, lr_rate, decay)

    # Print a model summary for visualization.
    print(model.summary())

    # Fit and train our model.
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

    # Evaluate the model on our test data.
    scores = evaluate_model(model, X_test, y_test)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


if __name__ == '__main__':
    # Config
    # Number of epochs to train with.
    epochs = 5
    # Learning rate for our SGD classifier.
    lr_rate = 0.01
    # Batch size to process.
    batch_size = 64
    # Decay rate.
    decay = lr_rate / epochs
    # Train model on GC. If set to False, model will train locally. Default=False.
    # Before training on GC, update: config.yaml, run_cloud.sh, load_data.py ->_DATA_DIR_CLOUD.
    cloud_train = True
    # Multi gpu support. Replace the below number with # of gpus. Default: gpus=0
    gpus = 0

    # Start model build, train, test, and predict process.
    dispatch()