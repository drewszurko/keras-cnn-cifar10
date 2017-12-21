from __future__ import print_function
import requests
from tqdm import tqdm
import tarfile
import os
import sys

_DATA_DIR_ORIGINAL = 'cifar-10-batches-py'
_DATA_DIR_LOCAL = 'cifar10-data'
_DATA_TAR = 'cifar-10-python.tar.gz'
_DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
_rel_path = os.path.dirname(os.path.realpath(sys.argv[0]))


# Creates http request to download a missing or corrupted dataset.
def download_data():
    file_path = _get_file_path(_DATA_TAR)

    print('Downloading %s.' % _DATA_DIR_LOCAL)

    # Create data download request.
    r = requests.get(_DATA_URL, stream=True)

    # Check for successful request.
    if r.status_code != 200:
        return print("\n %s could not be downloaded. Please try again." % _DATA_DIR_LOCAL)

    # Retrieve file size in bytes from header.
    total_size = int(r.headers.get('content-length', 0))

    # Writes file to 'data' directory while displaying a download progress bar in the users terminal.
    with open(file_path, 'wb') as f:
        for chunk in tqdm(iterable=r.iter_content(chunk_size=1024), total=int(total_size / 1024), unit='KB', ncols=100):
            f.write(chunk)

    # Untar the downloaded file.
    untar_data()


def untar_data():
    # Get rel dir paths for extracting data.
    org_path = _get_file_path(_DATA_DIR_ORIGINAL)
    tar_path = _get_file_path(_DATA_TAR)
    file_path = _get_file_path(_DATA_DIR_LOCAL)

    # Open the downloaded tar file.
    tar = tarfile.open(tar_path)

    # Untar into the current directory.
    tar.extractall(_rel_path)

    # Close file.
    tar.close()

    # Rename the directory.
    os.rename(org_path, file_path)

    # Discard the downloaded tar file from users computer.
    os.remove(tar_path)

    print("Download complete. Now let's clean up!")


def _get_file_path(filepath):
    # Join rel & filepath can get correct dir to download cifar10-data into.
    data_path = os.path.join(_rel_path, filepath)
    return data_path
