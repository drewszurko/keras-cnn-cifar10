from __future__ import print_function
from trainer.data_utils.download_util import download_data
import os
from tensorflow.python.lib.io import file_io
import pickle
import six
import numpy as np
import sys

# Do not manually change this variable. Cloud training should be specified from main.py.
is_cloud_train = False

# Local/remote location of cifar10 dataset.
_DATA_DIR_LOCAL = 'cifar10-data'
_DATA_DIR_CLOUD = 'gs://keras-ml-models/cifar10-data'
_DATA_BATCH = 'data_batch_'
_TEST_BATCH = 'test_batch'
_BATCH_META = 'batches.meta'

# Get real path so we can check if cifar10-data is already downloaded.
rel_path = os.path.dirname(os.path.realpath(sys.argv[0]))
data_path = os.path.join(rel_path, _DATA_DIR_LOCAL)


def load_dataset(cloud_train):
    # Set the global is_cloud_train variable so we can return the correct data file_path.
    global is_cloud_train
    is_cloud_train = cloud_train

    print("\nImporting data. Please wait...")

    # Check if cifar10-data directory already exists. If not, download data.
    if not cloud_train and not os.path.isdir(data_path):
        print("\nNo data found. Let's fix that!")
        download_data()

    # Load our train data.
    x_train, y_train = load_training_data()

    # Load our test data.
    x_test, y_test = load_test_data()

    return (x_train, y_train), (x_test, y_test)


def load_training_data():
    # Create image/class arrays.
    images = np.zeros(shape=[50000, 3, 32, 32], dtype=float)
    classes = np.zeros(shape=[50000], dtype=int)

    # Starting index for the current batch.
    index_start = 0

    # Process each data_batch file in the cifar10-data.
    for i in range(5):
        # Loads an image/class batch file from the cifar10-data directory.
        image_batch, class_batch = load_data(filename=_DATA_BATCH + str(i + 1))

        # Count of images in the batch.
        num_images = len(image_batch)

        # Ending index for the current batch.
        index_end = index_start + num_images

        # Store the batch images into the above images array.
        images[index_start:index_end, :] = image_batch

        # Store the batch class labels into the above classes array.
        classes[index_start:index_end] = class_batch

        # Generate the starting index for the next data_batch file.
        index_start = index_end

    return images, classes


def load_test_data():
    images, classes = load_data(_TEST_BATCH)
    return images, classes


def load_data(filename):
    # Load the pickled data_batch file.
    data = unpickle_file(filename)

    # Grab raw images.
    raw_images = data[b'data']

    # Grab the class identifiers for the raw images.
    classes = data[b'labels']

    # Convert identifying classes to numpy array.
    classes = np.array(classes)

    # Reshape our images for our model.
    images = reshape_images(raw_images)

    return images, classes


def unpickle_file(filename):
    # Retrieve the current file's file path.
    file_path = get_file_path(filename)

    # Load file. If Python 3, we must specify file encoding.
    with file_io.FileIO(file_path, mode='rb') as file:
        if six.PY3:
            data = pickle.load(file, encoding='bytes')
        else:
            data = pickle.load(file)
    return data


# Return correct filepath & filename for local or cloud training.
def get_file_path(filename=""):
    if not is_cloud_train:
        usr_dir = data_path
    else:
        usr_dir = _DATA_DIR_CLOUD
    return os.path.join(usr_dir, filename)


def reshape_images(raw_images):
    # Normalize the raw images from the batch files by converting them to floats.
    raw_float = np.array(raw_images, dtype=float) / 255.0

    # Reshape the raw_float array to 4 dimensions.
    images = raw_float.reshape([-1, 32, 32, 3])

    # Transpose image array indices.
    images = images.transpose([0, 3, 2, 1])

    return images
