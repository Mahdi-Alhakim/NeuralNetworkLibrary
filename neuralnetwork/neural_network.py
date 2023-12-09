import numpy as np

class NeuralNetwork:
    """
    Neural Network Library

    This module provides a simple implementation of feedforward neural networks.
    """

    def __init__(self, layer_sizes):
        """
        Initialize a neural network with given layer sizes.

        Parameters:
        - layer_sizes (list): List of integers representing the number of nodes in each layer.
        """
        self.num_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        # Initialize weights with random values using the Xavier/Glorot initialization
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i+1]) / np.sqrt(layer_sizes[i]) for i in range(self.num_layers - 1)]
        self.biases = [np.zeros((1, size)) for size in layer_sizes[1:]]
        self.layer_outputs = []

    def sigmoid(self, x):
        """
        Sigmoid activation function.

        Parameters:
        - x (numpy array): Input array.

        Returns:
        - numpy array: Output after applying the sigmoid activation.
        """
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """
        Derivative of the sigmoid activation function.

        Parameters:
        - x (numpy array): Input array.

        Returns:
        - numpy array: Derivative of the sigmoid activation.
        """
        return x * (1 - x)

    def forward(self, inputs):
        """
        Perform forward pass through the network.

        Parameters:
        - inputs (numpy array): Input data.

        Returns:
        - numpy array: Output after forward pass.
        """
        self.layer_outputs = [inputs]
        for i in range(self.num_layers - 1):
            inputs = self.sigmoid(np.dot(inputs, self.weights[i]) + self.biases[i])
            self.layer_outputs.append(inputs)
        return inputs

    def backward(self, inputs, targets, learning_rate):
        """
        Perform backward pass through the network and update weights and biases.

        Parameters:
        - inputs (numpy array): Input data.
        - targets (numpy array): Target output.
        - learning_rate (float): Learning rate for weight and bias updates.
        """
        output_error = targets - self.layer_outputs[-1]
        output_delta = output_error * self.sigmoid_derivative(self.layer_outputs[-1])

        for i in range(self.num_layers - 2, -1, -1):
            layer_error = output_delta.dot(self.weights[i].T)
            layer_delta = layer_error * self.sigmoid_derivative(self.layer_outputs[i])

            self.weights[i] += self.layer_outputs[i].T.dot(output_delta) * learning_rate
            self.biases[i] += np.sum(output_delta, axis=0, keepdims=True) * learning_rate

            output_delta = layer_delta

    def train_batch(self, training_inputs, training_targets, epochs, learning_rate):
        """
        Train the network using batch gradient descent.

        Parameters:
        - training_inputs (numpy array): Input data for training. Each row represents a data point.
        - training_targets (numpy array): Target outputs for training. Each row corresponds to the target for the corresponding input.
        - epochs (int): Number of training epochs.
        - learning_rate (float): Learning rate for weight updates.
        """
        for epoch in range(epochs):
            for inputs, targets in zip(training_inputs, training_targets):
                inputs = np.array(inputs, ndmin=2)
                targets = np.array(targets, ndmin=2)

                self.forward(inputs)
                self.backward(inputs, targets, learning_rate)

    def train_mini_batch(self, training_inputs, training_targets, batch_size, epochs, learning_rate):
        """
        Train the network using mini-batch gradient descent.

        Parameters:
        - training_inputs (numpy array): Input data for training.
        - training_targets (numpy array): Target outputs for training.
        - batch_size (int): Size of each mini-batch. Must be less than or equal to the number of training samples.
        - epochs (int): Number of training epochs.
        - learning_rate (float): Learning rate for weight updates.
        """
        num_samples = len(training_inputs)
        for epoch in range(epochs):
            indices = np.random.permutation(num_samples)
            for start in range(0, num_samples, batch_size):
                batch_indices = indices[start:start+batch_size]
                batch_inputs = training_inputs[batch_indices]
                batch_targets = training_targets[batch_indices]

                for inputs, targets in zip(batch_inputs, batch_targets):
                    inputs = np.array(inputs, ndmin=2)
                    targets = np.array(targets, ndmin=2)

                    self.forward(inputs)
                    self.backward(inputs, targets, learning_rate)

    def train_single(self, inputs, targets, learning_rate):
        """
        Train the network using stochastic gradient descent (single data point).

        Parameters:
        - inputs (numpy array): Input data for training.
        - targets (numpy array): Target outputs for training.
        - learning_rate (float): Learning rate for weight updates.
        """
        inputs = np.array(inputs, ndmin=2)
        targets = np.array(targets, ndmin=2)

        self.forward(inputs)
        self.backward(inputs, targets, learning_rate)

    def predict(self, inputs):
        """
        Make predictions using the trained network.

        Parameters:
        - inputs (numpy array): Input data for predictions.

        Returns:
        - numpy array: Predicted outputs.
        """
        inputs = np.array(inputs, ndmin=2)
        return self.forward(inputs)
        
    def mutate(self, mutation_rate):
        """
        Perform muutations on the neural network weights and biases.

        Parameters:
        - mutation_rate (float): Rate in which mutation occurs.
        """
        for i in range(len(self.weights)):
            mask = (np.random.rand(*self.weights[i].shape) < mutation_rate).astype(float)
            random_values = np.random.randn(*self.weights[i].shape)
            self.weights[i] += mask * random_values

            mask_biases = (np.random.rand(*self.biases[i].shape) < mutation_rate).astype(float)
            random_biases = np.random.randn(*self.biases[i].shape)
            self.biases[i] += mask_biases * random_biases

    def copy(self):
        """
        Makes a deep copy of the neural network.

        Returns:
        - NeuralNetwork: Copy of self.
        """
        nn = NeuralNetwork(self.layer_sizes)
        nn.weights = self.weights.copy()
        nn.biases = self.biases.copy()
        return nn

if __name__ == "__main__":
    # Example usage:
    input_size = 2
    hidden_sizes = [3, 4]
    output_size = 1

    # Initialize neural network
    nn = NeuralNetwork([input_size] + hidden_sizes + [output_size])

    # Training data
    training_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    training_targets = np.array([[1], [0], [0], [0]])

    # Batch training
    nn.train_batch(training_inputs, training_targets, epochs=10000, learning_rate=0.1)

    # Mini-Batch training
    nn.train_mini_batch(training_inputs, training_targets, batch_size=100, epochs=10000, learning_rate=0.1)

    # Single data training
    nn.train_single([0, 0], [0], learning_rate=0.1)

    # Make predictions
    predictions = nn.predict(training_inputs)
    print(predictions)

