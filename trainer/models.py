from __future__ import print_function
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.models import Sequential
from keras.layers import Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from keras.constraints import maxnorm
from keras.optimizers import SGD
import tensorflow as tf

K.set_image_dim_ordering('th')

_MODEL = 'model_'


def build_model(num_classes, gpus):
    # If gpus count >=2, return the multi_gpu_model.
    # 9+ gpus is not supported so Keras will throw an error prompting the user to enter valid GPU #.
    if gpus >= 2:
        with tf.device('/cpu:0'):
            # Create the model.
            model = _create_model(num_classes)
            # Wrap model in Keras multi_gpu_model for multi GPU support.
            return multi_gpu_model(model, gpus)
    else:
        # Other return regular model b/c it supports gpu counts <=1.
        return _create_model(num_classes)


def _create_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def compile_model(model, lrate, decay):
    opt = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model


def evaluate_model(model, x_test, y_test):
    scores = model.evaluate(x_test, y_test)
    return scores
