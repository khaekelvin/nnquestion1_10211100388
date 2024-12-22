import unittest
import numpy as np
from models.neural_network import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        self.nn = NeuralNetwork(input_size=3, hidden_size=4, output_size=1)
        self.X = np.array([[1, 2, 3], [4, 5, 6]])
        self.y = np.array([[10], [20]])

    def test_initialization(self):
        self.assertEqual(self.nn.input_size, 3)
        self.assertEqual(self.nn.hidden_size, 4)
        self.assertEqual(self.nn.output_size, 1)
        self.assertEqual(self.nn.W1.shape, (3, 4))
        self.assertEqual(self.nn.W2.shape, (4, 1))

    def test_forward_pass(self):
        output = self.nn.forward(self.X)
        self.assertEqual(output.shape, (2, 1))

    def test_training(self):
        initial_loss = np.mean((self.nn.forward(self.X) - self.y) ** 2)
        losses = self.nn.train(self.X, self.y, epochs=100, verbose=False)
        final_loss = losses[-1]
        self.assertLess(final_loss, initial_loss)

    def test_prediction_shape(self):
        prediction = self.nn.predict(np.array([[1, 2, 3]]))
        self.assertEqual(prediction.shape, (1, 1))

    def test_relu_activation(self):
        test_input = np.array([-1, 0, 1])
        expected_output = np.array([0, 0, 1])
        np.testing.assert_array_equal(self.nn.relu(test_input), expected_output)

if __name__ == '__main__':
    unittest.main()