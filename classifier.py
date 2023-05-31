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
    # currently sigmoid, (use softmax instead?), same for every layer
    return 1 / (1 + np.exp(-x))

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


def shallow_NN():
    # input layer size: 28^2 (amount of pixels)
    # hidden layer size: try different ones to prevent overfitting (why?) first try: 500 (so about 64%) (try out Grid search(hyperparameter optimization) later on)
    # output layer size: 10 (classification to number)
    learning_rate = .1
    il_size = 28 ** 2
    hl_size = 500
    ol_size = 10

    cur_img = images[0]
    cur_label = labels[0]

    # one more for easy bias calculation
    input_layer = np.array(cur_img + [1])
    weights_in_hid = np.ones((hl_size + 1, il_size + 1))
    weights_hid_out = np.ones((ol_size, hl_size + 1))

    # forward
    hidden_layer_old = np.matmul(weights_in_hid, input_layer)
    hidden_layer = vec(relu, hidden_layer_old)
    output_layer_old = np.matmul(weights_hid_out, hidden_layer)
    output_layer = vec(relu, output_layer_old)

    # backpropagation TODO
    """
    we have C(R(Who*R(Wih*input) as the cost function of our entire net
    we want derivative wrt Who / Wih using chain rule
    let Who[1] denote the first row
    we have (C' o R o A)(Who[1]) where o is function composition and A(x) := x * R(Wih*input)
    wrt Who[i]: C'(R(Who[i]*R(Wih*input))) * R'(Who[i] * R(Wih*input)) * R(Wih*input)
    wrt Wih: C'(R(Who*R(Wih[i]*input))) * R'(Who * R(Wih[i]*input)) * Who * R'(Wih[i]*input) * input
    """
    Eo = []
    for i in range(weights_hid_out.shape[0]):
        n = weights_hid_out[i, :] @ hidden_layer
        m = 1 if i == cur_label else 0
        Eo.append((relu(n) - m) * relu_derivative(n))
    Eo = np.reshape(np.array(Eo), (-1, 1))
    Eh = []
    for i in range(weights_in_hid.shape[0]):
        n = weights_in_hid[i, :] @ output_layer
        m = 1 if i == cur_label else 0

    der_who = Eo @ np.reshape(hidden_layer, (1, -1))
    der_wih = Eh @ input_layer

    #update weights
    weights_hid_out -= learning_rate * der_who
    weights_in_hid -= learning_rate * der_wih


shallow_NN()
