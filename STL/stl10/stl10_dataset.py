from __future__ import print_function

import contextlib
import sys
import os, sys, tarfile, errno
import numpy as np
import matplotlib.pyplot as plt
import cv2


if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib  # ugly but works
else:
    import urllib

try:
    from imageio import imsave
except Exception:
    from scipy.misc import imsave


# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the directory with the data
DATA_DIR = "/datasets"

# url of the binary data
DATA_URL = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"

# path to the binary train file with image data
DATA_PATH_TRAN = f"{DATA_DIR}/stl10_binary/train_X.bin"

# path to the binary train file with labels
LABEL_PATH_TRAN = f"{DATA_DIR}/stl10_binary/train_y.bin"

# path to the binary test file with image data
DATA_PATH_TEST = f"{DATA_DIR}/stl10_binary/test_X.bin"

# path to the binary train file with labels
LABEL_PATH_TEST = f"{DATA_DIR}/stl10_binary/test_y.bin"


def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, "rb") as f:
        return np.fromfile(f, dtype=np.uint8)


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, "rb") as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def read_single_image(image_file):
    """
    CAREFUL! - this method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    image = np.transpose(image, (2, 1, 0))
    return image


def plot_image(image):
    """
    :param image: the image to be plotted in a 3-D matrix format
    :return: None
    """
    plt.imshow(image)
    plt.show()


def save_image(image, name):
    imsave(f"{name}.png", image, format="png")


def download_and_extract():
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
        tarfile.open(filepath, "r:gz").extractall(dest_directory)


def save_tran_images(images, labels):
    print("Saving images to disk")
    i = 0
    for image in images:
        label = labels[i]
        directory = f"{DATA_DIR}/tran_img/{str(label)}/"
        with contextlib.suppress(OSError):
            os.makedirs(directory, exist_ok=True)
        filename = directory + str(i)
        print(filename)
        save_image(image, filename)
        i = i + 1

def set_dir(dir):
    global DATA_DIR
    DATA_DIR = f'/{dir}'

def save_test_images(images, labels):
    print("Saving images to disk")
    i = 0
    for image in images:
        label = labels[i]
        directory = f"{DATA_DIR}/test_img/{str(label)}/"
        with contextlib.suppress(OSError):
            os.makedirs(directory, exist_ok=True)
        filename = directory + str(i)
        print(filename)
        save_image(image, filename)
        i = i + 1


def build():
    # download data if needed
    download_and_extract()

    # test to check if the image is read correctly
    with open(DATA_PATH_TEST) as f:
        image = read_single_image(f)
        plot_image(image)

    # test to check if the whole dataset is read correctly
    test_images = read_all_images(DATA_PATH_TEST)
    print(test_images.shape)

    test_labels = read_labels(LABEL_PATH_TEST)
    print(test_labels.shape)

    # save images to disk
    save_test_images(test_images, test_labels)

    # test to check if the whole dataset is read correctly
    test_images = read_all_images(DATA_PATH_TEST)
    print(test_images.shape)

    test_labels = read_labels(LABEL_PATH_TEST)
    print(test_labels.shape)

    # save images to disk
    save_test_images(test_images, test_labels)

    with open(DATA_PATH_TRAN) as f:
        image = read_single_image(f)
        plot_image(image)

    # test to check if the whole dataset is read correctly
    tran_images = read_all_images(DATA_PATH_TRAN)
    print(tran_images.shape)

    tran_labels = read_labels(LABEL_PATH_TRAN)
    print(tran_labels.shape)

    # save images to disk
    save_tran_images(tran_images, tran_labels)

    # test to check if the whole dataset is read correctly
    tran_images = read_all_images(DATA_PATH_TRAN)
    print(tran_images.shape)

    tran_labels = read_labels(LABEL_PATH_TRAN)
    print(tran_labels.shape)

    # save images to disk
    save_tran_images(tran_images, tran_labels)


def load_data():
    tran_data = f"{DATA_DIR}/tran_img"
    test_data = f"{DATA_DIR}/test_img"
    clases = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]

    traning_data = []  # create training data

    def create_training_data():
        for cate in clases:
            path = os.path.join(tran_data, cate)
            class_num = clases.index(cate)
            for img in os.listdir(path):
                with contextlib.suppress(Exception):
                    img_array = cv2.imread(
                        os.path.join(path, img), cv2.IMREAD_GRAYSCALE
                    )
                    new_array = cv2.resize(img_array, (96, 96))
                    traning_data.append([new_array, class_num])

    testing_data = []  # create test and valid data

    def create_testing_data():
        for cate in clases:
            path = os.path.join(test_data, cate)
            class_num = clases.index(cate)
            for img in os.listdir(path):
                with contextlib.suppress(Exception):
                    img_array = cv2.imread(
                        os.path.join(path, img), cv2.IMREAD_GRAYSCALE
                    )
                    new_array = cv2.resize(img_array, (96, 96))
                    testing_data.append([new_array, class_num])

    create_testing_data()
    create_training_data()

    X_train = []  # feature ->X
    y_train = []  # label   ->y
    X_test = []
    y_test = []

    def stor(x, y, z):
        for feature, label in z:
            x.append(feature)
            y.append(label)

    stor(X_train, y_train, traning_data)
    stor(X_test, y_test, testing_data)

    return (X_train, y_train), (X_test, y_test)


def class_name():
    return [
        "airplane",
        "bird",
        "car",
        "cat",
        "deer",
        "dog",
        "horse",
        "monkey",
        "ship",
        "truck",
    ]



