from preproc import *

def vec(func, iterable):
    return np.fromiter(map(lambda x: func(x), iterable), float)

def vec_der(func, iterable):
    return np.fromiter(map(lambda x: func(x, derivative = True), iterable), float)

class Layer:
    def __init__(self, size, activation, bias=True):
        self.bias = bias
        bias_node = 1 if self.bias else 0
        self.nodes = np.zeros(size + bias_node)
        self.nodes_pre = np.zeros(size + bias_node)
        self.activation = activation


    def set_nodes(self, content):
        self.nodes = np.array([1] + list(content.flatten())) if self.bias else content


    def __str__(self):
        str_nodes = self.nodes[1:] if self.bias else self.nodes
        bias = f" bias: {self.nodes[0]}" if self.bias else ""
        return "[" + ", ".join(f"{el}" for el in str_nodes) + "]" + bias


    def __len__(self):
        return len(self.nodes)


    def activate(self):
        self.nodes_pre = self.nodes.copy()
        self.nodes = np.array([1] + list(vec(self.activation, self.nodes.flatten()[1:]))) if self.bias else vec(self.activation, self.nodes.flatten())


    def get_nodes(self):
        return self.nodes[1:] if self.bias else self.nodes


    def get_nodes_pre(self):
        return self.nodes_pre[1:] if self.bias else self.nodes_pre


def sigmoid(x, derivative=False):
    if derivative:
        return np.exp(-x)/((1 + np.exp(-x))**2)
    return 1/(1 + np.exp(-x))



def relu(x, derivative=False):
    if derivative:
        return 0 if x <= 0 else 1 # non-differentiable in 0 ~> problem?
    return max(0, x)


def cost(classification, wanted, derivative=False): # TODO read about cross entropy
    dif = wanted - classification
    if derivative:
        return -dif/len(dif)#1/len(dif) * dif
    return 1/(2*len(dif)) * np.dot(dif, dif)#1/(2 * len(dif)) * np.dot(dif, dif)


def softmax(x, derivative=False):
    if derivative:
        pass
    # numerical stabilization
    new_x = x - max(x)
    return np.exp(new_x) / np.sum(np.exp(new_x))


class Neural_net:
    """
    TODO:
    change activation functions of layers (default sigmoid?)
    currently wants at least one hidden layer, change?
    try out types?
    change constructor parameters, confusing rn
    rename content to nodes as in layer
    sigmoid derivative is sigmoid * (1 - sigmoid), so no need to calculate exp again - change
    """
    def __init__(self, learning_rate, num_hid, il_size, ol_size, *hl_sizes):
        """
        num_hid: amount of hidden layers
        il_size: amount of input nodes
        ol_size: amount of output nodes
        args: nodes per hidden layer
        """

        if len(hl_sizes) != num_hid:
            raise TypeError("please specify hidden layer sizes")

        self.layers = [Layer(il_size, relu)] + list(Layer(size, relu) for size in hl_sizes) + [Layer(ol_size, softmax, bias = False)]
        # make bias column/row 1 instead of 0
        self.weights = [np.hstack((np.reshape(np.ones(hl_sizes[0]), (-1, 1)), np.zeros((hl_sizes[0], il_size))))] \
            + list(np.hstack((np.reshape(np.ones(hl_sizes[i+1], np.zeros((hl_sizes[i+1], hl_sizes[i]))), (-1, 1)))) for i in range(num_hid - 1)) \
            + [np.hstack((np.reshape(np.ones(ol_size), (-1, 1)), np.zeros((ol_size, hl_sizes[-1]))))]

        self.learning_rate = learning_rate


    def set_input_layer(self, nodes):
        if len(nodes) != len(self.layers[0]) - 1:
            raise TypeError("wrong amount of nodes")

        self.layers[0].set_nodes(nodes)


    def show_output_layer(self):
        print(self.layers[-1])


    def forward(self):
        for layer in range(len(self.layers) - 1):
            self.layers[layer+1].set_nodes(self.weights[layer] @ self.layers[layer].nodes)
            self.layers[layer+1].activate()


    def backpropagate(self, output):
        """
        output: what the output layer should have been
        """
        derivatives = []
        # last layer
        vec = vec_der(softmax, self.layers[-1].get_nodes_pre().flatten())
        cost_grad = cost(self.layers[-1].get_nodes(), output, derivative = True)
        derivatives.append(np.reshape(vec * cost_grad, (-1, 1)) @ np.reshape(self.layers[-2].get_nodes(), (1, -1)))

        # earlier layers
        for layer in range(len(self.layers) - 2, 0, -1):
            # calculate cost gradient wrt activated nodes
            dc_dz = vec_der(relu, self.layers[layer+1].get_nodes_pre().flatten()) * cost_grad
            cost_grad = np.fromiter((np.dot(self.weights[layer][:, k], dc_dz) for k in range(1, len(self.layers[layer]))), float) # maybe 1 in range wrong
            #nodes = self.layers[layer].get_nodes()
            #vec = nodes * (1 - nodes) # nodes_pre wrong? !!!!!!!!!!!!!!!!!!!!!!!!!!
            vec = vec_der(relu, self.layers[layer].nodes_pre[1:])
            derivatives.append(np.reshape(vec * cost_grad, (-1, 1)) @ np.reshape(self.layers[layer-1].get_nodes(), (1, -1)))

        # update weights
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.hstack((np.zeros((self.weights[i].shape[0], 1)), derivatives[len(derivatives) - (i + 1)]))


    def save_weights(self):
        pass #TODO
    #np.save("who.npy", weights_hid_out)
    #np.save("wih.npy", weights_in_hid)




def main():
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

    # input layer size: 28^2 (amount of pixels)
    # hidden layer size: try different ones to prevent overfitting (why?) first try: 500 (so about 64%) (try out Grid search(hyperparameter optimization) later
    # output layer size: 10 (classification to number)
    learning_rate = .01
    il_size = 28 ** 2
    hl_size = 100
    ol_size = 10
    net = Neural_net(learning_rate, 1, il_size, ol_size, hl_size)

    print("training model...")

    retrain = True

    if retrain:
        costs = []
        for (image, label, num) in zip(images, labels, range(len(labels))):
            net.set_input_layer(np.array(image)/255) # normalize input to [0,1]
            net.forward()
            wanted = np.zeros(10)
            wanted[label] = 1
            net.backpropagate(wanted)
            costs.append(cost(net.layers[-1].get_nodes(), wanted))
            if num == 5000:
                break

        print("training done")


    else:
        weights_hid_out = np.load("who.npy")
        weights_in_hid = np.load("wih.npy")

    plt.plot(costs)
    plt.show()

    print("loading test data")
    img_fileinfo = read_image_file(files[2])
    images = img_fileinfo["images"]
    label_fileinfo = read_label_file(files[3])
    labels = label_fileinfo["labels"]
    print("finished")

    print("testing model...")
    num_correct = 0
    for (image, label, num) in zip(images, labels, range(len(labels))):
        net.set_input_layer(np.array(image)/255)
        net.forward()
        output = net.layers[-1].get_nodes()
        wanted = np.zeros(10)
        wanted[label] = 1
        highest = list(output).index(max(output))
        if highest == label:
            num_correct += 1
        if num == 3000:
            break

    print(num_correct / 3000)




if __name__ == "__main__":
    main()

# CARE that later the layers have numbers in a good range, otherwise derivative of sigmoid function very small -> very slow learning!
