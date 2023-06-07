from preproc import *


def vec(func, iterable):
    return np.fromiter(map(lambda x: func(x), iterable), float)


def vec_der(func, iterable):
    return np.fromiter(map(lambda x: func(x, derivative=True),
                           iterable), float)


class Layer:
    def __init__(self, size, activation, bias=True):
        self.bias = bias
        bias_node = 1 if self.bias else 0
        self.nodes = np.zeros(size + bias_node)
        self.nodes_pre = np.zeros(size + bias_node)
        self.activation = activation

    def set_nodes(self, content):
        self.nodes = np.array([1] + list(content.flatten())) \
                if self.bias else content

    def __str__(self):
        str_nodes = self.nodes[1:] if self.bias else self.nodes
        bias = f" bias: {self.nodes[0]}" if self.bias else ""
        return "[" + ", ".join(f"{el}" for el in str_nodes) + "]" + bias

    def __len__(self):
        return len(self.nodes)

    def activate(self):
        self.nodes_pre = self.nodes.copy()
        if not self.bias:
            self.nodes = self.activation(self.nodes)
            return
        if self.bias:
            activated = vec(self.activation, self.nodes.flatten()[1:])
            self.nodes = np.array([1] + list(activated))
        else:
            self.nodes = vec(self.activation, self.nodes.flatten())

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
        return 0 if x <= 0 else 1
    return max(0, x)


def cost(classification, wanted, derivative=False):
    dif = wanted - classification
    if derivative:
        return -dif/len(dif)
    return 1/(2 * len(dif)) * np.dot(dif, dif)


def softmax(x, derivative=False):
    if derivative:
        jacobian = np.outer(x, -x) + np.diag(x)
        return jacobian
    # numerical stabilization
    new_x = x - max(x)
    return np.exp(new_x) / np.sum(np.exp(new_x))


def cat_cross_entropy(classification, wanted):
    return -np.dot(wanted, np.log(classification))


class Neural_net:
    """
    TODO:
    change activation functions of layers (default relu?)
    currently wants at least one hidden layer, change?
    try out types?
    change constructor parameters, confusing rn
    sigmoid derivative is sigmoid * (1 - sigmoid),
        so no need to calculate exp again - change
    """
    def __init__(self, learning_rate, num_hid, il_size, ol_size, *hl_sizes):
        """
        num_hid: amount of hidden layers
        il_size: amount of input nodes
        ol_size: amount of output nodes
        hl_sizes: nodes per hidden layer
        """

        if len(hl_sizes) != num_hid:
            raise TypeError("please specify hidden layer sizes")

        input_layer = Layer(il_size, relu)
        hidden_layers = list(Layer(size, relu) for size in hl_sizes)
        output_layer = Layer(ol_size, softmax, bias=False)
        self.layers = [input_layer] + hidden_layers + [output_layer]

        self.weights = []
        max_size = max(il_size, ol_size, *hl_sizes)
        interval_size = 2 / np.sqrt(max_size)
        for size1, size2 in zip([il_size, *hl_sizes], [*hl_sizes, ol_size]):
            bias_weights = np.reshape(np.ones(size2), (-1, 1))
            weight_matrix = np.random.rand(size2, size1)
            # squish weights into wanted range
            weight_matrix = interval_size * (weight_matrix - 0.5)
            # add bias weights for bias calculation
            weight_matrix = np.hstack((bias_weights, weight_matrix))
            self.weights.append(weight_matrix)

        self.learning_rate = learning_rate
        self.derivatives = [np.zeros(weight.shape) for weight in self.weights]

    def set_input_layer(self, nodes):
        if len(nodes) != len(self.layers[0]) - 1:
            raise TypeError("wrong amount of nodes")

        self.layers[0].set_nodes(nodes)

    def show_output_layer(self):
        print(self.layers[-1])

    def gradient_check(self, inp, outp):
        cur_weights = self.weights.copy()
        epsilon = 0.0001
        # check 100 pseudo random gradients
        for _ in range(100):
            k = np.random.randint(0, len(self.weights))
            i, j = np.random.randint(0, self.weights[k].shape[0]), \
                np.random.randint(1, self.weights[k].shape[1])
            self.set_input_layer(inp)
            self.forward()
            self.backpropagate(outp)
            calculated = self.derivatives[k][i, j]
            self.weights[k][i, j] += epsilon
            self.set_input_layer(inp)
            self.forward()
            plus = cost(self.layers[-1].get_nodes(), outp)
            self.weights[k][i, j] -= epsilon
            self.set_input_layer(inp)
            self.forward()
            minus = cost(self.layers[-1].get_nodes(), outp)
            numerical_approx = (plus - minus) / (2 * epsilon)
            if (d := abs(calculated - numerical_approx)) > 0.01:
                relative_dif = d / max(abs(calculated), abs(numerical_approx))
                print(relative_dif, calculated, numerical_approx)
        self.set_input_layer(inp)
        self.forward()
        self.backpropagate(outp)
        cur_error = self.errors[-1]
        for k in range(len(cur_error)):
            calculated = cur_error[k]
            self.layers[-1].nodes = self.layers[-1].nodes_pre
            self.layers[-1].nodes[k] -= epsilon
            self.layers[-1].activate()
            minus = cost(self.layers[-1].get_nodes(), outp)
            self.layers[-1].nodes = self.layers[-1].nodes_pre
            self.layers[-1].nodes[k] += epsilon
            self.layers[-1].activate()
            plus = cost(self.layers[-1].get_nodes(), outp)
            numerical_approx = (plus - minus) / (2 * epsilon)
            print(abs(calculated - numerical_approx) / max(
                abs(calculated), abs(numerical_approx)),
                        calculated, numerical_approx)

    def forward(self):
        for layer in range(len(self.layers) - 1):
            self.layers[layer+1].set_nodes(
                    self.weights[layer] @ self.layers[layer].nodes)
            self.layers[layer+1].activate()

    def backpropagate(self, output):
        """
        output: what the output layer should have been
        """
        self.errors = []

        # calculate gradient of cost function wrt last layer before activation
        error = self.layers[-1].get_nodes() - output
        # this is specifically for softmax with categorical cross entropy
        self.errors.append(error)

        for layer in range(len(self.layers) - 2, 0, -1):

            # calculate gradient wrt activation of earlier layer
            local_grad_act = self.weights[layer][:, 1:]
            grad_act = local_grad_act.T @ self.errors[-1]
            # calculate gradient wrt pre activation layer
            grad_pre = vec_der(
                    relu, self.layers[layer].get_nodes_pre()) * grad_act

            self.errors.append(grad_pre)

        self.errors.reverse()

        # calculate loss derivatives wrt weights
        for num, (layer, err) in enumerate(zip(self.layers[:-1], self.errors)):
            weight_derivatives = np.outer(err, layer.get_nodes())
            bias_derivatives = np.reshape(err, (-1, 1))
            derivative_matrix = np.hstack((bias_derivatives,
                                           weight_derivatives))
            self.derivatives[num] += derivative_matrix

    def update_weights_and_biases(self):
        for weight, derivative in zip(self.weights, self.derivatives):
            weight -= self.learning_rate * derivative
        self.derivatives = [np.zeros(weight.shape) for weight in self.weights]

    def mini_batch_gd(inputs, outputs):
        if len(inputs) != len(outputs):
            raise TypeError("#input not equal to #outputs")

        for inp, outp in zip(inputs, outputs):
            self.forward(inp)
            self.backpropagate(outp)

        self.update_weights_and_biases()

    def save_weights(self):
        pass  # TODO
    # np.save("who.npy", weights_hid_out)
    # np.save("wih.npy", weights_in_hid)


def main():
    files = ["train-images-idx3-ubyte", "train-labels-idx1-ubyte",
             "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"]

    print("loading training data...")
    tr_img_fileinfo = read_image_file(files[0])
    images = tr_img_fileinfo["images"]
    tr_label_fileinfo = read_label_file(files[1])
    labels = tr_label_fileinfo["labels"]
    print("finished")

    # view_img =
    # lambda num: view_image(images[num], tr_img_fileinfo["num_rows"],
    #                                  tr_img_fileinfo["num_cols"])
    # view_img(99)
    # print(labels[99])

    # input layer size: 28^2 (amount of pixels)
    # hidden layer size: try different ones to prevent overfitting (why?)
    # first try: 500 (so about 64%)
    # (try out Grid search(hyperparameter optimization) later
    # output layer size: 10 (classification to number)

    learning_rate = .0001
    il_size = 28 ** 2
    hl_size = 500
    ol_size = 10
    net = Neural_net(learning_rate, 1, il_size, ol_size, hl_size)

    print("training model...")

    retrain = True

    if retrain:
        costs = []
        images = images[:5000]
        labels = labels[:5000]
        c = 0
        for num, (image, label) in enumerate(zip(images, labels)):
            if num == 10:
                wanted2 = np.zeros(10)
                wanted2[label] = 1
                net.gradient_check(np.array(image)/255, wanted2)
                exit()
            net.set_input_layer(np.array(image)/255)
            net.forward()
            wanted = np.zeros(10)
            wanted[label] = 1
            net.backpropagate(wanted)
            c += cat_cross_entropy(net.layers[-1].get_nodes(), wanted)
            net.update_weights_and_biases()

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
    for num, (image, label) in enumerate(zip(images, labels)):
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
