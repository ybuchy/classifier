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
#view_img(99)
#print(labels[99])

def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 0 if x <= 0 else 1

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return np.exp(x)/((1 + np.exp(x))**2)

def cost(classification, label):
    # ppl use cross entropy, maybe try this first
    wanted = np.zeros(10)
    wanted[label] = 1
    dif = classification - wanted
    return 1/2 * np.dot(dif, dif)

def cost_derivative(classification, label):
    wanted = np.zeros(10)
    wanted[label] = 1
    return classification - wanted

def vec(func, iterable):
    return np.fromiter(map(lambda x: func(x), iterable), float)


# TODO
def forward():
    pass

# TODO
def backward():
    pass

def shallow_NN():
    # input layer size: 28^2 (amount of pixels)
    # hidden layer size: try different ones to prevent overfitting (why?) first try: 500 (so about 64%) (try out Grid search(hyperparameter optimization) later on)
    # output layer size: 10 (classification to number)
    learning_rate = .1
    il_size = 28 ** 2
    hl_size = 500
    ol_size = 10

    print("training model...")
    weights_in_hid = np.random.rand(hl_size + 1, il_size + 1)
    weights_hid_out = np.random.rand(ol_size, hl_size + 1)

    retrain = False

    if retrain:

        for num in range(len(images)):
            print(f"cur at {num} of {len(images)}")
            cur_img = images[num]
            cur_label = labels[num]

            # one more for easy bias calculation
            input_layer = np.array(cur_img + [1])

            # forward
            hidden_layer_old = weights_in_hid @ input_layer
            # using sigmoid rn because otherwise numbers get too big to calculate sigmoid derivative for delta_2
            hidden_layer = vec(sigmoid, hidden_layer_old)
            output_layer_old = weights_hid_out @ hidden_layer
            output_layer = vec(sigmoid, output_layer_old)

            # backpropagation
            wanted = np.zeros(10)
            wanted[cur_label] = 1
            delta_2 = (output_layer - wanted) * vec(sigmoid_derivative, output_layer_old)
            # TODO changes bias????????
            der_who = []
            for err in delta_2:
                der_who.append(err * hidden_layer)
            der_who = np.array(der_who)
            weights_no_bias = weights_hid_out[:, :-1]
            delta_1 = vec(sigmoid_derivative, hidden_layer[:-1]) * (np.transpose(weights_no_bias) @ delta_2)
            der_wih = []
            for err in delta_1:
                der_wih.append((err * input_layer))
            der_wih = np.vstack((der_wih, np.zeros(il_size + 1)))

            #update weights
            weights_hid_out -= learning_rate * der_who
            weights_in_hid -= learning_rate * der_wih

        print("done training")

        np.save("who.npy", weights_hid_out)
        np.save("wih.npy", weights_in_hid)

    else:
        weights_hid_out = np.load("who.npy")
        weights_in_hid = np.load("wih.npy")

    img_fileinfo = read_image_file(files[2])
    imgs = img_fileinfo["images"]
    label_fileinfo = read_label_file(files[3])
    lbls = label_fileinfo["labels"]
    print("finished")

    print("testing model...")
    num_correct = 0
    for num in range(len(imgs)):
        input_layer = np.array(imgs[num] + [1])
        hidden_layer_old = weights_in_hid @ input_layer
        hidden_layer = vec(sigmoid, hidden_layer_old)
        output_layer_old = weights_hid_out @ hidden_layer
        output_layer = vec(sigmoid, output_layer_old)
        highest = list(output_layer).index(max(output_layer))
        if highest == lbls[num]:
            num_correct += 1

    print(num_correct / len(imgs))



shallow_NN()
