import unittest
import torch
from torch import nn
from cl_module.classifier import *

class TestNN(unittest.TestCase):
    def test_cat_cross_entropy_batchtensor(self):
        inp_a, inp_b = np.random.rand(50), np.random.rand(50)
        ind_a, ind_b = np.random.randint(0, 50), np.random.randint(0, 50)
        out_a, out_b = np.zeros(50), np.zeros(50)
        out_a[ind_a], out_b[ind_b] = 1, 1
        cl_tensor = np.hstack((np.reshape(inp_a, (-1, 1)),
                               np.reshape(inp_b, (-1, 1))))
        print(cl_tensor.shape)
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
        net.forward(net_tensor)
        clas_fw = net.layers[-1].get_units()
        # compare
        np.testing.assert_allclose(clas_fw[:,0], torch_fw1, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(clas_fw[:,1], torch_fw2, rtol=1e-5, atol=1e-6)

    def test_batch_backprop_relu_softmax(self):
        pass


if __name__ == "__main__":
    unittest.main()
