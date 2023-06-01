from preproc import *

class Layer:
    def __init__(self, size, activation, bias=True):
        self.bias = bias
        bias_node = 1 if self.bias else 0
        self.nodes = np.zeros(size + bias_node)


    def set_nodes(self, content):
        self.nodes = [1] + content if self.bias else content


    def __str__(self):
        str_nodes = self.nodes[1:] if self.bias else self.nodes
        bias = f" bias: {self.nodes[0]}" if self.bias else ""
        return "[" + ", ".join(f"{el}" for el in str_nodes) + "]" + bias


    def __len__(self):
        return len(self.nodes)

class Neural_net:
    """
    TODO:
    change activation functions of layers (default sigmoid?)
    currently wants at least one hidden layer, change?
    """
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))


    def __init__(self, num_hid, il_size, ol_size, *hl_sizes):
        """
        num_hid: amount of hidden layers
        il_size: amount of input nodes
        ol_size: amount of output nodes
        args: nodes per hidden layer
        """

        if len(hl_sizes) != num_hid:
            raise TypeError("please specify hidden layer sizes")

        self.layers = [Layer(il_size, self.sigmoid)] + list(Layer(size, self.sigmoid) for size in hl_sizes) + [Layer(ol_size, self.sigmoid, bias = False)]
        self.weights = [np.zeros((hl_sizes[0], il_size))] + list(np.zeros(hl_sizes[i+1], hl_sizes[i]) for i in range(num_hid - 1)) + [np.zeros((ol_size, hl_sizes[-1]))]


    def set_input_layer(self, nodes):
        if len(nodes) != len(self.layers[0]) - 1:
            raise TypeError("wrong amount of nodes")

        self.layers[0].set_nodes(nodes)


    def forward(self):
        for layer in self.layers:
            print(layer)
        for layer in range(len(self.layers) - 1):
            self.layers[layer+1].set_nodes(self.layers[layer] @ self.weights[layer])



test = Neural_net(1, 1, 1, 1)
test.set_input_layer(np.array([1]))
test.forward()
exit()

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
    return np.exp(-x)/((1 + np.exp(-x))**2)

def cost(classification, label):
    # ppl use cross entropy, maybe try this first
    wanted = np.zeros(10)
    wanted[label] = 1
    dif = classification - wanted
    return 1/(2 * len(dif)) * np.dot(dif, dif)

def cost_derivative(classification, label):
    wanted = np.zeros(10)
    wanted[label] = 1
    dif = classification - wanted
    return 1/len(dif) * dif

def vec(func, iterable):
    return np.fromiter(map(lambda x: func(x), iterable), float)


# TODO
def forward():
    pass

# TODO
def backward():
    pass

def main():
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
    # CARE that later the layers have numbers in a good range, otherwise derivative of sigmoid function very small -> very slow learning!
    weights_in_hid = np.zeros((hl_size + 1, il_size + 1))
    weights_hid_out = np.zeros((ol_size, hl_size + 1))

    print(weights_hid_out.shape)

    retrain = True

    if retrain:

        for num in range(len(images)):
            print(f"cur at {num} of {len(images)}")
            cur_img = images[num]
            cur_label = labels[num]

            # the [1] will be for bias, same later for hidden layer
            input_layer = np.array(cur_img + [1])

            # forward
            hidden_layer_old = weights_in_hid @ input_layer
            # using sigmoid rn because otherwise numbers get too big to calculate sigmoid derivative for delta_2
            hidden_layer = vec(relu, hidden_layer_old) + [1]
            output_layer_old = weights_hid_out @ hidden_layer
            output_layer = vec(sigmoid, output_layer_old)
            # backpropagation
            wanted = np.zeros(10)
            wanted[cur_label] = 1
            c = 1/ol_size * (output_layer - wanted)
            sd = vec(sigmoid_derivative, output_layer_old)
            a = np.reshape(c * sd, (-1, 1))
            der_who = a @ np.reshape(hidden_layer, (1, -1))
            """
            for j in range(j_max):
                for k in range(k_max):
                    der_who[j, k] = a[j] * hidden_layer[k]
            """
            # TODO refactor into function, then try on small NN
            sd = vec(sigmoid_derivative, hidden_layer_old)
            B = 1/ol_size * (output_layer - wanted) * vec(relu_derivative, output_layer_old)
            s = B @ weights_hid_out
            der_wih = np.reshape(sd * s, (-1, 1)) @ np.reshape(input_layer, (1, -1))
            """
            for j in range(j_max):
                sd = sigmoid_derivative(hidden_layer_old[j])
                s =  sum(weights_hid_out[l, j] * sigmoid_derivative(output_layer_old[l]) * 1/ol_size * (output_layer[l] - wanted[l]) for l in range(ol_size))
                for k in range(k_max):
                    der_wih[j, k] = input_layer[k] * sd[j] * s[j]
            """

            #update weights
            weights_hid_out -= learning_rate * der_who
            weights_in_hid -= learning_rate * der_wih


        print("training done")

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
        print(output_layer)
        highest = list(output_layer).index(max(output_layer))
        if highest == lbls[num]:
            num_correct += 1

    print(num_correct / len(imgs))




if __name__ == "__main__":
    main()
