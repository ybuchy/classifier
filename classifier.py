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
        # ToDO !!!!!!!!!!!!!!!!! change !!!!!!!!!!!!!!!
        if not self.bias:
            self.nodes = softmax(self.nodes)
            return
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
    sigmoid derivative is sigmoid * (1 - sigmoid), so no need to calculate exp again - change
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

        self.layers = [Layer(il_size, relu)] + list(Layer(size, relu) for size in hl_sizes) + [Layer(ol_size, softmax, bias = False)]


        max_size = il_size # change

        self.weights = [np.hstack((np.reshape(np.ones(hl_sizes[0]), (-1, 1)), 2 / np.sqrt(max_size) * (np.random.rand(hl_sizes[0], il_size) - 0.5)))] \
            + list(np.hstack((np.reshape(np.ones(hl_sizes[i+1]), 2 / np.sqrt(max_size) * (np.random.rand(hl_sizes[i+1], hl_sizes[i]) - 0.5)), (-1, 1))) for i in range(num_hid - 1)) \
            + [np.hstack((np.reshape(np.ones(ol_size), (-1, 1)), 2 / np.sqrt(max_size) * (np.random.rand(ol_size, hl_sizes[-1]) - 0.5)))]

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
            i, j = np.random.randint(0, self.weights[k].shape[0]), np.random.randint(1, self.weights[k].shape[1])
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
            numerical_approx = (plus - minus) / 2 * epsilon
            if((d := abs(calculated - numerical_approx)) > 0.01):
                print(d / max(abs(calculated), abs(numerical_approx)), calculated, numerical_approx)

        return
        # check random errors TODO !!!!!!!!!!!!!!!!!!!!!!!!
        self.set_input_layer(inp)
        self.forward()
        cur_errors = self.errors.copy()
        print(len(cur_errors))
        exit()
        for _ in range(10):
            k = np.random.randint(0, len(cur_errors))
            j = np.random.randint(0, len(cur_errors[k]))
            calculated = cur_errors[k][j]
            self.layer[k][j] -= epsilon
            # forward





    def forward(self):
        for layer in range(len(self.layers) - 1):
            self.layers[layer+1].set_nodes(self.weights[layer] @ self.layers[layer].nodes)
            self.layers[layer+1].activate()


    def backpropagate(self, output):
        """
        output: what the output layer should have been
        """
        self.errors = []

        # calculate gradient of cost function wrt last layer before activation
        error = self.layers[-1].get_nodes() - output # this is specifically for softmax with categorical cross entropy right now
        self.errors.append(error)

        for layer in range(len(self.layers) - 2, 0, -1):

            # calculate gradient wrt activation of earlier layer
            local_act_grad = self.weights[layer][:, 1:]
            grad_act =  local_act_grad.T @ self.errors[-1]
            # calculate gradient wrt pre activation layer
            grad_pre = vec_der(relu, self.layers[layer].get_nodes_pre()) * grad_act

            self.errors.append(grad_pre)

        print(self.errors)

        # calculate loss derivatives wrt weights
        for (layer, error, num) in zip(self.layers[:-1], reversed(self.errors), range(len(self.weights))):
            self.derivatives[num] += np.hstack((np.reshape(error, (-1, 1)), np.outer(error, layer.get_nodes())))


    def update_weights_and_biases(self):
        for (weight, derivative) in zip(self.weights, self.derivatives):
            weight -= self.learning_rate * derivative
        self.derivatives = [np.zeros(weight.shape) for weight in self.weights]


    def mini_batch_gd(inputs, outputs):
        if len(inputs) != len(outputs):
            raise TypeError("#input not equal to #outputs")

        for (inp, outp) in zip(inputs, outputs):
            self.forward(inp)
            self.backpropagate(outp)

        self.update_weights_and_biases()



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
        for (image, label, num) in zip(images, labels, range(len(labels))):
            if num  == 10:
                wanted2 = np.zeros(10)
                wanted2[label] = 1
                net.gradient_check(np.array(image)/255, wanted2)
                exit()
            net.set_input_layer(np.array(image)/255) # normalize input to [0,1]
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
