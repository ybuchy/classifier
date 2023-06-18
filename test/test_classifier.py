import unittest
import torch
from torch import nn
from cl_module.classifier import *

class TestNN(unittest.TestCase):
    def test_relu_batch_layer_forward(self):
        relu = ReLU_layer(3, 2, 2)
        inp = np.array([[1, 2, 3], [3, 2, 1]]).T
        relu.bias = np.array([1., 4.])
        relu.weight = np.array([[3, 4, 5], [-6, -7, -8]])

        outp = relu.forward(inp)

        np.testing.assert_equal(outp, np.array([[27, 0], [23, 0]]).T)

    def test_relu_batch_layer_gradient_check(self):
        pass

    def test_relu_batch_layer_pre_grad(self):
        relu = ReLU_layer(3, 2, 2)
        inp = np.array([[1, 2, 3], [3, 2, 1]]).T
        relu.bias = np.array([1., 4.])
        relu.weight = np.array([[3, 4, 5], [-6, -7, -8]])

        relu.forward(inp)

        np.testing.assert_equal(relu.calc_inp_grad(np.ones((2, 2))), np.array([[3, 4, 5], [3, 4, 5]]).T)

    def test_relu_batch_layer_weight_grad(self):
        relu = ReLU_layer(3, 2, 2)
        inp = np.array([[1, 2, 3], [3, 2, 1]]).T
        relu.bias = np.array([1., 4.])
        relu.weight = np.array([[3, 4, 5], [-6, -7, -8]])

        outp = relu.forward(inp)

        np.testing.assert_equal(relu.calc_weight_grad(np.ones((2, 2))), np.array([[4, 4, 4], [0, 0, 0]]).T)

    def test_relu_batch_layer_bias_grad(self):
        relu = ReLU_layer(3, 2, 2)
        inp = np.array([[1, 2, 3], [3, 2, 1]]).T
        relu.bias = np.array([1., 4.])
        relu.weight = np.array([[3, 4, 5], [-6, -7, -8]])

        relu.forward(inp)

        np.testing.assert_equal(relu.calc_bias_grad(np.ones((2, 2))), np.array([2, 0]).T)

    def test_sigmoid_batch_layer_forward(self):
        pass

    def test_sigmoid_batch_layer_gradient_check(self):
        pass

    def test_cat_cross_entropy_batchtensor(self):
        inp_a, inp_b = np.random.rand(50), np.random.rand(50)
        ind_a, ind_b = np.random.randint(0, 50), np.random.randint(0, 50)
        out_a, out_b = np.zeros(50), np.zeros(50)
        out_a[ind_a], out_b[ind_b] = 1, 1
        cl_tensor = np.hstack((np.reshape(inp_a, (-1, 1)),
                               np.reshape(inp_b, (-1, 1))))
        lb_tensor = np.hstack((np.reshape(out_a, (-1, 1)),
                               np.reshape(out_b, (-1, 1))))

        a = cat_cross_entropy(inp_a, out_a)
        b = cat_cross_entropy(inp_b, out_b)
        tensor = cat_cross_entropy_batchtensor(cl_tensor, lb_tensor)

        np.testing.assert_allclose(np.array((a, b)), tensor, rtol=1e-5, atol=1e-6)

    def test_softmax_batchtensor(self):
        inp_a, inp_b = np.random.rand(50), np.random.rand(50)
        inp_tensor = np.hstack((np.reshape(inp_a, (-1, 1)), np.reshape(inp_b, (-1, 1))))

        a = softmax(inp_a)
        b = softmax(inp_b)
        tensor = softmax_batchtensor(inp_tensor)

        np.testing.assert_allclose(np.array((a, b)).T, tensor, rtol=1e-5, atol=1e-6)
        

    def test_relu_batchtensor(self):
        tensor = np.array([[0, 1, -1], [-1, 0, 1]])
        out = np.array([[0, 1, 0], [0, 0, 1]])

        relu_tensor = relu_batchtensor(tensor)

        np.testing.assert_equal(relu_tensor, out)


    def test_batch_forward_relu_softmax(self):
        return
        input_size, hidden_size, output_size = 50, 30, 10
        # pytorch model to test against
        model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=0))
        # classifier model
        net = NN_classifier(0, 2, 1, input_size, output_size, hidden_size)
        # instantiate same weights
        for i in range(2):
            bias = model[2*i].bias.detach().numpy().reshape((-1, 1))
            weights = model[2*i].weight.detach().numpy()
            weight_bias_matrix = np.hstack((bias, weights))
            net.weights[i] = weight_bias_matrix
        # get random inputs
        inp1, inp2 = torch.rand(50), torch.rand(50)
        # classifier input batch matrix
        net_tensor = np.hstack((np.reshape(inp1.detach().numpy(), (-1, 1)),
                                np.reshape(inp2.detach().numpy(), (-1, 1))))

        # do forward
        torch_fw1 = model.forward(inp1).detach().numpy()
        torch_fw2 = model.forward(inp2).detach().numpy()
        clas_fw = net.forward(net_tensor)

        # compare
        np.testing.assert_allclose(clas_fw[:,0], torch_fw1, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(clas_fw[:,1], torch_fw2, rtol=1e-5, atol=1e-6)

    def test_batch_backprop_relu_softmax_gradient_check(self):
        pass
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


if __name__ == "__main__":
    unittest.main()
