import os, sys
import urllib.request as urllib

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3
# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH
# path to the directory with the data
DATA_DIR = "./data"
# url of the binary data
DATA_URL = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"


def download_dataset():
    """
    Download and extract the STL-10 dataset
    :return: None
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split("/")[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                "\rDownloading %s %.2f%%"
                % (filename, float(count * block_size) / float(total_size) * 100.0)
            )
            sys.stdout.flush()

        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print("Downloaded", filename)

download_dataset()
