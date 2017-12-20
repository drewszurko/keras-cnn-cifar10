from __future__ import print_function
import requests
from tqdm import tqdm
import tarfile
import os

_DATA_DIR_ORIGINAL = 'cifar-10-batches-py'
_DATA_DIR_LOCAL = 'cifar10-data'
_DATA_TAR = 'cifar-10-python.tar.gz'
_DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'


# Creates http request to download a missing or corrupted dataset.
def download_data():
    print('Downloading %s.' % _DATA_DIR_LOCAL)

    # Create data download request.
    r = requests.get(_DATA_URL, stream=True)

    # Check for successful request.
    if r.status_code != 200:
        return print("\n %s could not be downloaded. Please try again." % _DATA_DIR_LOCAL)

    # Retrieve file size in bytes from header.
    total_size = int(r.headers.get('content-length', 0))

    # Writes file to 'data' directory while displaying a download progress bar in the users terminal.
    with open(_DATA_TAR, 'wb') as f:
        for chunk in tqdm(iterable=r.iter_content(chunk_size=1024), total=int(total_size / 1024), unit='KB', ncols=100):
            f.write(chunk)

    # Untar the downloaded file.
    untar_data()


def untar_data():
    # Open the downloaded tar file.
    tar = tarfile.open(_DATA_TAR)

    # Untar into the current directory.
    tar.extractall()

    # Close file.
    tar.close()

    # Rename the directory.
    os.rename(_DATA_DIR_ORIGINAL, _DATA_DIR_LOCAL)

    # Discard the downloaded tar file from users computer.
    os.remove(_DATA_TAR)

    print("Download complete. Now let's clean up!")
