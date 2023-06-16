import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from PIL import Image
#from typing import TypeAlias
from collections.abc import Iterable

# Own types for type hinting
#image: TypeAlias = npt.NDArray[np.int_]
#label: TypeAlias = npt.NDArray[np.int_]
# fileinfo ~> own class?
#fileinfo: TypeAlias = dict[str, str | int | list[image]]
#image_batch: TypeAlias = list[image]
#label_batch: TypeAlias = list[label]


class Preproc:
    """
    image files pixel values are 0 (background / white)
    to 255 (foreground / black)
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    a0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel

     label files (labels are 0 to 9)
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label

    def view_image(img, num_rows, num_cols):
        im = Image.new('L', (num_rows, num_cols))
        im.putdata(img)
        plt.imshow(np.asarray(im), cmap='gray_r', vmin=0, vmax=255)
        plt.show()

    view_img =
    lambda num: view_image(images[num], tr_img_fileinfo["num_rows"],
                                     tr_img_fileinfo["num_cols"])
    view_img(99)
    print(labels[99])
    """

    @staticmethod
    def read_image_file(filename):
#    def read_image_file(filename: str) -> fileinfo:
        with open(filename, "rb") as f:
            b = f.read()

            file_info = {}
            file_info["magic_number_hex"] = b[:4].hex()
            file_info["num_images"] = int(b[4:8].hex(), 16)
            file_info["num_rows"] = int(b[8:12].hex(), 16)
            file_info["num_cols"] = int(b[12:16].hex(), 16)
            file_info["images"] = []
            offset = 16

            for img_num in range(file_info["num_images"]):
                img_size = file_info["num_rows"] * file_info["num_cols"]
                image = b[offset + img_num * img_size:
                          offset + (img_num + 1) * img_size].hex()
                image = np.array([int(image[i:i+2], 16)
                                  for i in range(0, len(image), 2)])
                # input normalization for classifier
                image = 1/255 * image
                file_info["images"].append(image)

        return file_info

    @staticmethod
    def read_label_file(filename):
#    def read_label_file(filename: str) -> fileinfo:
        with open(filename, "rb") as f:
            b = f.read()
            file_info = {}
            file_info["magic_number_hex"] = b[:4].hex()
            file_info["num_labels"] = int(b[4:8].hex(), 16)
            file_info["labels"] = []

            offset = 8
            for label_num in range(file_info["num_labels"]):
                label = b[offset + label_num]
                # add label as one-hot encoded vector
                lb = np.zeros(10)
                lb[label] = 1
                file_info["labels"].append(lb)

        return file_info

    @staticmethod
    def generate_batches(images, labels, batch_size):
#    def generate_batches(images: Iterable[image],
#                         labels: Iterable[label],
#                         batch_size: int) -> tuple[list[image_batch], list[label_batch]]:
        img_batches = []
        label_batches = []
        for offset in range(len(images) // batch_size):
            cur_ind = offset * batch_size
            img_batches.append(images[cur_ind:cur_ind+batch_size].T)
            label_batches.append(labels[cur_ind:cur_ind+batch_size].T)
        if (ind := len(images) // batch_size * batch_size) != len(images):
            img_batches.append(images[ind:].T)
            label_batches.append(labels[ind:].T)
        return img_batches, label_batches

    def get_training_set(self, batch_size):
    #def get_training_set(self, batch_size: int) -> list[tuple[image_batch, label_batch]]:
        if not self.loaded:
            raise AttributeError("mnist data not in memory yet, use load() first")
        val_size = int(1/5 * self.tr_set_file_info["num_images"])
        tr_images = np.array(self.tr_set_file_info["images"])
        tr_labels = np.array(self.tr_label_file_info["labels"])
        tr_images = tr_images[:-val_size]
        tr_labels = tr_labels[:-val_size]
        img_batches, label_batches = self.generate_batches(
            tr_images, tr_labels, batch_size)
        return list(zip(img_batches, label_batches))

    def get_validation_set(self, batch_size):
#    def get_validation_set(self, batch_size: int) -> list[tuple[image_batch, label_batch]]:
        if not self.loaded:
            raise AttributeError("mnist data not in memory yet, use load() first")
        val_size = int(1/5 * self.tr_set_file_info["num_images"])
        test_images = np.array(self.tr_set_file_info["images"])
        test_labels = np.array(self.tr_label_file_info["labels"])
        val_images = test_images[-val_size:]
        val_labels = test_labels[-val_size:]
        img_batches, label_batches = self.generate_batches(
            val_images, val_labels, batch_size)
        return list(zip(img_batches, label_batches))

    def get_test_set(self, batch_size):
        #def get_test_set(self, batch_size: int) -> list[tuple[image_batch, label_batch]]:
        if not self.loaded:
            raise AttributeError("mnist data not in memory yet, use load() first")
        test_images = np.array(self.test_set_file_info["images"])
        test_labels = np.array(self.test_label_file_info["labels"])
        img_batches, label_batches = self.generate_batches(
            test_images, test_labels, batch_size)
        return list(zip(img_batches, label_batches))

    def load(self) -> None:

        self.tr_set_file_info = self.read_image_file(self.tr_file)
        self.tr_label_file_info = self.read_label_file(self.tr_label_file)
        self.test_set_file_info = self.read_image_file(self.test_file)
        self.test_label_file_info = self.read_label_file(self.test_label_file)
        self.loaded = True

    def __init__(self, tr_set: str, tr_labels: str, test_set: str, test_label: str) -> None:
        self.tr_file = tr_set
        self.tr_label_file = tr_labels
        self.test_file = test_set
        self.test_label_file = test_label
        self.loaded = False
