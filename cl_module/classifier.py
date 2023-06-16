import numpy as np


"""
def vec(func, iterable):
    return np.fromiter(map(lambda x: func(x), iterable), float)
"""


class Layer:
    @staticmethod
    def add_bias(unit_matrix):
        return np.vstack((np.ones(unit_matrix.shape[1]), unit_matrix))

    def __init__(self, shape, activation, bias=True):
        self.bias = bias
        bias_node = 1 if self.bias else 0
        self.units = np.zeros(shape)
        if bias:
            self.units = self.add_bias(self.units)
        self.activation = activation

    def set_units(self, units):
        if self.bias:
            units = self.add_bias(units)
        self.units = units

    def __str__(self):
        return np.array2string(self.units)

    def __len__(self):
        return len(self.units)

    def activate(self):
        self.units_pre = self.units.copy()
        if not self.bias:
            self.units = self.activation(self.units)
        else:
            activated = self.activation(self.units[1:, :])
            self.units = self.add_bias(activated)

    def get_units(self):
        return self.units[1:, :] if self.bias else self.units

    def get_units_pre(self):
        return self.units_pre[1:, :] if self.bias else self.units_pre


def sigmoid(x, derivative=False):
    if derivative:
        return np.exp(-x)/((1 + np.exp(-x))**2)
    return 1/(1 + np.exp(-x))


def relu(x, derivative=False):
    if derivative:
        return np.vectorize(lambda num: 0 if num <= 0 else 1)(x)
    return np.vectorize(lambda num: max(0, num))(x)


def relu_batchtensor(x):
    return np.vectorize(lambda num: max(0, num))(x)


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


def softmax_batchtensor(x):
    new_x = x - np.amax(x, axis=1)
    # numerical stabilization
    e_new_x = np.exp(new_x)
    return e_new_x / np.sum(e_new_x, axis=1)


def cat_cross_entropy(classification, wanted):
    return -np.inner(wanted, np.log(classification))


def cat_cross_entropy_batchtensor(classification, wanted):
    return -np.diag(wanted @ np.log(classification).T)


class NN_classifier:
    """
    Neural net for multi-class classification problems.
    Currently uses softmax on output layer with categorical cross entropy as loss function
    for hidden layers, the default is relu right now
    """
    """
    TODO:
    change activation functions of layers (default relu?)
    currently wants at least one hidden layer, change?
    try out types?
    change constructor parameters, confusing rn
    sigmoid derivative is sigmoid * (1 - sigmoid),
        so no need to calculate exp again - change
    rename nodes to units
    !!!!!!!!!!!!!!!!!!!!!![important TODOs]!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    whole batch into one matrix ~> layer not vector but matrix DONE (debug now)
    start using validation set
    write unittests ~> use for batch debugging?
    do forward(input), set input layer feels kinda useless
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    implement L2 regularization
    implement dropout regularization
    implement adam or similar for learning rate
    """
    def __init__(self, learning_rate, batch_size, num_hid, il_size, ol_size, *hl_sizes):
        """
        num_hid: amount of hidden layers
        il_size: amount of input nodes
        ol_size: amount of output nodes
        hl_sizes: nodes per hidden layer
        """

        if len(hl_sizes) != num_hid:
            raise TypeError("please specify hidden layer sizes")

        input_layer = Layer((il_size, batch_size), relu)
        hidden_layers = list(Layer((size, batch_size), relu) for size in hl_sizes)
        output_layer = Layer((ol_size, batch_size), softmax_batchtensor, bias=False)
        self.layers = [input_layer] + hidden_layers + [output_layer]

        self.weights = []
        for size1, size2 in zip([il_size, *hl_sizes], [*hl_sizes, ol_size]):
            bias_weights = np.reshape(np.zeros(size2), (-1, 1))
            weight_matrix = np.random.rand(size2, size1)
            # squash weights into wanted range
            weight_matrix = np.sqrt(2 / size1) * (weight_matrix - 0.5)
            # add bias weights for bias calculation
            weight_matrix = np.hstack((bias_weights, weight_matrix))
            self.weights.append(weight_matrix)

        self.learning_rate = learning_rate
        self.derivatives = [np.zeros(weight.shape) for weight in self.weights]

    def set_input_layer(self, units):
        self.layers[0].set_units(units)

    def show_output_layer(self):
        print(self.layers[-1])

    """
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
            self.weights = cur_weights.copy()
            self.weights[k][i, j] += epsilon
            self.set_input_layer(inp)
            self.forward()
            plus = cat_cross_entropy(self.layers[-1].get_nodes(), outp)
            self.weights = cur_weights.copy()
            self.weights[k][i, j] -= epsilon
            self.set_input_layer(inp)
            self.forward()
            minus = cat_cross_entropy(self.layers[-1].get_nodes(), outp)
            numerical_approx = (plus - minus) / (2 * epsilon)
            if (d := abs(calculated - numerical_approx)) > 0.01:
                relative_dif = d / max(abs(calculated), abs(numerical_approx))
                print(k, i, j, relative_dif, calculated, numerical_approx)
    """

    def forward(self):
        for layer in range(len(self.layers) - 1):
            self.layers[layer+1].set_units(
                    self.weights[layer] @ self.layers[layer].units)
            self.layers[layer+1].activate()

    def backpropagate(self, output):
        """
        output: what the output layer should have been
        """
        self.errors = []

        # calculate gradient of cost function wrt last layer before activation
        error = self.layers[-1].get_units() - output
        # this is specifically for softmax with categorical cross entropy
        self.errors.append(error)

        for layer in range(len(self.layers) - 2, 0, -1):

            # calculate gradient wrt activation of earlier layer
            local_grad_act = self.weights[layer][:, 1:]
            grad_act = local_grad_act.T @ self.errors[-1]
            # calculate gradient wrt pre activation layer
            grad_pre = relu(self.layers[layer].get_units_pre(), derivative=True) * grad_act

            self.errors.append(grad_pre)

        self.errors.reverse()

        # calculate loss derivatives wrt weights
        derivatives = []
        for num, (layer, err) in enumerate(zip(self.layers[:-1], self.errors)):
            weight_derivatives = err @ layer.get_units().T # not sure!, was np.outer(err, layer.get_nodes())
            bias_derivatives = np.reshape(np.sum(err, axis=1), (-1, 1)) # not sure!
            derivative_matrix = np.hstack((bias_derivatives,
                                           weight_derivatives))
            derivatives.append(derivative_matrix)

        for weight, derivative in zip(self.weights, derivatives):
            weight -= self.learning_rate * derivative

        #def update_weights_and_biases(self):
        #self.derivatives = [np.zeros(weight.shape) for weight in self.weights]

    def mini_batch_gd(self, inp, outp):
        self.set_input_layer(inp)
        self.forward()
        self.backpropagate(outp)
        loss = cat_cross_entropy(self.layers[-1].get_units(), outp)
        #self.update_weights_and_biases()
        return sum(loss) / len(loss)

    def save_weights(self):
        pass  # TODO
    # np.save("who.npy", weights_hid_out)
    # np.save("wih.npy", weights_in_hid)
