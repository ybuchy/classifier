import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# image files pixel values are 0 (background / white) to 255 (foreground / black)
# [offset] [type]          [value]          [description]
#0000     32 bit integer  0x00000803(2051) magic number
#0004     32 bit integer  60000            number of images
#a0008     32 bit integer  28               number of rows
#0012     32 bit integer  28               number of columns
#0016     unsigned byte   ??               pixel
#0017     unsigned byte   ??               pixel
#........
#xxxx     unsigned byte   ??               pixel

# label files (labels are 0 to 9)
#[offset] [type]          [value]          [description]
#0000     32 bit integer  0x00000801(2049) magic number (MSB first)
#0004     32 bit integer  10000            number of items
#0008     unsigned byte   ??               label
#0009     unsigned byte   ??               label
#........
#xxxx     unsigned byte   ??               label
def view_image(img, num_rows, num_cols):
    im = Image.new('L', (num_rows, num_cols))
    im.putdata(img)
    plt.imshow(np.asarray(im), cmap='gray_r', vmin=0, vmax=255)
    plt.show()

def read_image_file(filename):
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
            image = b[offset + img_num * img_size : offset + (img_num + 1) * img_size].hex()
            image = [int(image[i:i+2], 16) for i in range(0, len(image), 2)]
            file_info["images"].append(image)
            
        return file_info

def read_label_file(filename, rload=False):
    with open(filename, "rb") as f:
        b = f.read()
        file_info = {}
        file_info["magic_number_hex"] = b[:4].hex()
        file_info["num_labels"] = int(b[4:8].hex(), 16)
        file_info["labels"] = []

        offset = 8
        for label_num in range(file_info["num_labels"]):
            label = b[offset + label_num]
            file_info["labels"].append(label)
            
        return file_info
