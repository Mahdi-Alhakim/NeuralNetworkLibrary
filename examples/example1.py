from neuralnetwork.neural_network import NeuralNetwork
import numpy as np

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
    print( "Training Data:\n    " + "\n    ".join([str(x)+" -> "+str(y) for x, y in zip(training_inputs, training_targets)]) )

    # Batch training
    nn.train_batch(training_inputs, training_targets, epochs=10000, learning_rate=0.1)

    # Mini-Batch training
    nn.train_mini_batch(training_inputs, training_targets, batch_size=100, epochs=10000, learning_rate=0.1)

    # Single data training
    nn.train_single([0, 0], [0], learning_rate=0.1)

    # Make predictions
    predictions = nn.predict(training_inputs)
    print( "\nOutputs:\n    " + "\n    ".join([str(x)+" -> "+str(y) for x, y in zip(training_inputs, predictions)]) )
