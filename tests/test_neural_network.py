# tests/test_neural_network.py
import unittest
import numpy as np
from neuralnetwork.neural_network import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def setUp(self):
        # Initialize a small neural network for testing
        self.input_size = 2
        self.hidden_sizes = [3, 4]
        self.output_size = 1
        self.nn = NeuralNetwork([self.input_size] + self.hidden_sizes + [self.output_size])

    def test_forward(self):
        # Test the forward pass
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        output = self.nn.forward(inputs)
        self.assertEqual(output.shape, (4, 1))

    def test_backward(self):
        # Test the backward pass
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        targets = np.array([[1], [0], [0], [0]])
        self.nn.forward(inputs)
        self.nn.backward(inputs, targets, learning_rate=0.1)

    def test_train_batch(self):
        # Test batch training
        training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        training_targets = np.array([[1], [0], [0], [0]])
        self.nn.train_batch(training_inputs, training_targets, epochs=100, learning_rate=0.1)

    def test_train_mini_batch(self):
        # Test mini-batch training
        training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        training_targets = np.array([[1], [0], [0], [0]])
        self.nn.train_mini_batch(training_inputs, training_targets, batch_size=2, epochs=100, learning_rate=0.1)

    def test_train_single(self):
        # Test single data point training
        self.nn.train_single([0, 0], [0], learning_rate=0.1)

    def test_predict(self):
        # Test prediction
        inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        predictions = self.nn.predict(inputs)
        self.assertEqual(predictions.shape, (4, 1))

    def test_mutate(self):
        # Test mutation
        original_weights = self.nn.weights.copy()
        original_biases = self.nn.biases.copy()
        self.nn.mutate(mutation_rate=0.1)

    def test_copy(self):
        # Test copying
        copied_nn = self.nn.copy()
        self.assertEqual(self.nn.layer_sizes, copied_nn.layer_sizes)

if __name__ == '__main__':
    unittest.main()