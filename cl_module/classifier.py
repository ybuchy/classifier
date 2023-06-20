import numpy as np


class Lin_batch_layer:
    @staticmethod
    def add_bias(unit_matrix): # Is that correct?
        return np.vstack((np.ones(unit_matrix.shape[1]), unit_matrix))

    def __init__(self, inp_size, outp_size, batch_size, bias=True):
        self.inp_size = inp_size
        self.outp_size = outp_size
        self.unit_matrix = np.zeros((inp_size, batch_size))
        self.outp_matrix = None # leave None?
        self.bias = np.zeros(outp_size) if bias else None # good idea to mix types?
        self.weight = 2 / inp_size * np.random.rand(outp_size, inp_size)
        self.bias_grad = np.zeros(outp_size)
        self.weight_grad = np.zeros(self.weight.shape)

    def forward(self, inp_matrix):
        self.unit_matrix = inp_matrix
        if not self.bias is None:
            inp_matrix = self.add_bias(inp_matrix)
        weight_bias_matrix = np.hstack((np.reshape(self.bias, (-1, 1)), self.weight)) # have weigth_bias_matrix as attribute, otherwise have to do this a million times!
        self.outp_matrix_pre = weight_bias_matrix @ inp_matrix
        self.outp_matrix = self.activate(self.outp_matrix_pre)
        return self.outp_matrix

    # @@@@@@@@@ The following all return sum of gradients over given batch
    def calc_inp_grad(self, output_grad):
        if self.outp_matrix is None:
            raise AttributeError("Calculate output first (obj.forward(...))")
        lin_grad = self.calc_lin_grad(output_grad)
        inp_grad = self.weight.T @ lin_grad
        return inp_grad

    def calc_gradients(self, output_grad):
        inp_grad = self.calc_inp_grad(output_grad)
        self.calc_bias_grad(output_grad)
        self.calc_weight_grad(output_grad)
        return inp_grad
        

    def calc_weight_grad(self, output_grad):
        if self.outp_matrix is None:
            raise AttributeError("Calculate output first (obj.forward(...))")
        lin_grad = self.calc_lin_grad(output_grad)
        weight_grad = lin_grad @ self.unit_matrix.T
        self.weight_grad += weight_grad

    def calc_bias_grad(self, output_grad):
        lin_grad = self.calc_lin_grad(output_grad)
        self.bias_grad += np.sum(lin_grad, axis=1)

    def update_parameters(self, lr):
        self.weight -= lr * self.weight_grad
        self.bias -= lr * self.bias_grad

    """
    def set_bias(self, bias_vector):
        self.bias = bias_vector

    def set_weigth

    def set_units(self, unit_matrix):
        # TODO check for correct shape
        self.unit_matrix = unit_matrix

    def get_units(self):
        return self.unit_matrix

    def get_outp_pre(self):
        return self.output_matrix_pre
    """

    def __str__(self):
        return np.array2string(self.unit_matrix)

    def __len__(self):
        return self.inp_size


class ReLU_layer(Lin_batch_layer):
    def __init__(self, inp_size, outp_size, batch_size, bias=True):
        super().__init__(inp_size, outp_size, batch_size, bias)

    def calc_lin_grad(self,  output_grad):
        outp = self.outp_matrix_pre
        if outp is None:
            raise AttributeError("Calculate output first (obj.forward(...))")
        lin_grad = np.maximum(np.zeros(outp.shape), np.sign(outp)) # TODO is there a better way to write this?
        return lin_grad * output_grad
        
    def activate(self, unit_matrix):
        return np.maximum(np.zeros(unit_matrix.shape), unit_matrix)


# TODO @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Linear_layer(Lin_batch_layer):
    def __init__(self, inp_size, outp_size, batch_size, bias=True):
        super().__init__(inp_size, outp_size, batch_size, bias)

    def calc_lin_grad(self, output_grad):
        return output_grad

    def activate(self, unit_matrix):
        return unit_matrix


class Sigmoid_layer(Lin_batch_layer):
    def __init__(self, inp_size, outp_size, batch_size, bias=True):
        super().__init__(inp_size, outp_size, batch_size, bias)

    def calc_lin_grad(self, output_grad):
        outp = self.output_matrix
        if outp is None:
            raise AttributeError("Calculate output first (obj.forward(...))")
        lin_grad = outp * (1 - outp)
        return lin_grad * output_grad

    def activate(self, unit_matrix):
        return 1 / (1 + np.exp(-x))


class Softmax_loss:
    def __init__(self, size, batch_size):
        self.unit_matrix = np.zeros((size, batch_size))

    def activate(self, unit_matrix):
        # numerical stabilization
        m = unit_matrix - np.amax(unit_matrix, axis=0)
        e_m = np.exp(m)
        self.act_matrix = e_m / np.sum(e_m, axis=0)

    def calc_lin_grad(self, one_hot_label_batch):
        return self.act_matrix - one_hot_label_batch

    def loss(self, one_hot_label_batch):
        self.activate(self.unit_matrix)
        classification = self.act_matrix
        return -np.diag(one_hot_label_batch.T @ np.log(classification))
        


"""
def sigmoid(x, derivative=False):
    if derivative:
        return np.exp(-x)/((1 + np.exp(-x))**2)
    return 1/(1 + np.exp(-x))
"""


# TODO interesting, works sometimes but not always (just returns all zeros matrix if it doesnt) @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
"""
def relu(x, derivative=False):
    if derivative:
        return np.vectorize(lambda num: 0 if num <= 0 else 1)(x)
    return np.vectorize(lambda num: max(0, num))(x)
"""


def relu_batchtensor(x, derivative=False):
    if derivative:
        return np.maximum(np.zeros(x.shape), np.sign(x)) # TODO is there a better way to write this?
    return np.maximum(np.zeros(x.shape), x)


"""
def cost(classification, wanted, derivative=False):
    dif = wanted - classification
    if derivative:
        return -dif/len(dif)
    return 1/(2 * len(dif)) * np.dot(dif, dif)
"""


def softmax(x, derivative=False):
    if derivative:
        jacobian = np.outer(x, -x) + np.diag(x)
        return jacobian
    # numerical stabilization
    new_x = x - max(x)
    return np.exp(new_x) / np.sum(np.exp(new_x))


def softmax_batchtensor(x):
    new_x = x - np.amax(x, axis=0)
    # numerical stabilization
    e_new_x = np.exp(new_x)
    return e_new_x / np.sum(e_new_x, axis=0)


def cat_cross_entropy(classification, wanted):
    return -np.inner(wanted, np.log(classification))


def cat_cross_entropy_batchtensor(classification, wanted):
    return -np.diag(wanted.T @ np.log(classification))


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
    !!!!!!!!!!!!!!!!!!!!!![important TODOs]!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    refactor (want layers way more modular)
        ideas: have layer(inp_shape, outp_shape) then do layer.calc_inp_grad(out_grad) / layer.calc_weight_grad(out_grad)
               how to model that layer is output_layer ~> dont have out_grad
        (ideas) should be way better to write tests for ~> way more robust. Also easier to add cnn tensor layers? (hard to do efficient calculations? ~ check np tensor)
        how to model bias? just put in layer?
    write more tests
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    implement adam or similar for learning rate
    implement L2 regularization
    implement dropout regularization
    """
    def __init__(self, learning_rate, batch_size):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.layers = []

    def add_layer(self, act, inp_size, outp_size=None):
        if act == "relu":
            self.layers.append(
                ReLU_layer(inp_size, outp_size, self.batch_size))
        elif act == "sigmoid":
            self.layers.append(
                Sigmoid_layer(inp_size, outp_size, self.batch_size))
        elif act == "softmax_loss":
            self.layers.append(
                Softmax_loss(inp_size, self.batch_size))

        elif act == "linear":
            self.layers.append(
                Linear_layer(inp_size, outp_size, self.batch_size))


    def add_relu(self, inp_size, outp_size):
        self.add_layer("relu", inp_size, outp_size)

    def add_sigmoid(self, inp_size, outp_size):
        self.add_layer("sigmoid", inp_size, outp_size)

    def add_softmax_loss(self, size):
        self.add_layer("softmax_loss", size)

    def add_linear(self, inp_size, outp_size):
        self.add_layer("linear", inp_size, outp_size)

    def show_output_layer(self):
        print(self.layers[-1])

    # TODO just hacking it to work rn
    def forward(self, inp_tensor):
        for layer in self.layers[:-1]:
            inp_tensor = layer.forward(inp_tensor)
        self.layers[-1].activate(inp_tensor)

    # cur hacky - only for softmax loss
    def backpropagate(self, output):
        # TODO is all the slicing inefficient?
        grad = self.layers[-1].calc_lin_grad(output)
        for layer in reversed(self.layers[:-1]):
            # TODO - inp_grad useless for input layer
            grad = layer.calc_gradients(grad)
        for layer in self.layers[:-1]:
            layer.update_parameters(self.learning_rate)

    def mini_batch_gd(self, inp, outp):
        self.forward(inp)
        self.backpropagate(outp)
        # pretty hacky rn
        loss = self.layers[-1].loss(outp)
        return sum(loss) / len(loss)

    def check_acc(self, ch_set):
        im_tensor, op_tensor = ch_set
        self.forward(im_tensor)
        output_layer = self.layers[-1].unit_matrix
        classifications = np.argmax(output_layer, axis=0)
        labels = np.argmax(op_tensor, axis=0)
        correct = 0
        for classification, label in zip(classifications, labels):
            if classification == label:
                correct += 1
        return correct / len(classifications)

    def save_weights(self):
        pass  # TODO
    # np.save("who.npy", weights_hid_out)
    # np.save("wih.npy", weights_in_hid)
