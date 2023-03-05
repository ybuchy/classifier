from preproc import *

files = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte", "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]

print("loading training data...")
tr_img_fileinfo = read_image_file(files[0])
images = tr_img_fileinfo["images"]
tr_label_fileinfo = read_label_file(files[1])
labels = tr_label_fileinfo["labels"]
print("finished")

view_img = lambda num: view_image(images[num], tr_img_fileinfo["num_rows"],
                                    tr_img_fileinfo["num_cols"])
view_img(100)
print(labels[100])
