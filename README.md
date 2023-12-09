# Neural Networks Library

A libaray for simple and easy implementation of customizable neural networks into projects.


## Installation:

1. To install the library, run:

``` bash
pip install .
```
OR
``` bash
python3 -m pip install .
```

## Execution:

#### To use the library, import it as follows:

``` python
 from neuralnetwork.neural_network import NeuralNetwork
```

## Unit Testing:

Unit tests performed were successful.

Run the available unit tests with:
``` bash
python3 -m unittest tests.test_neural_network
```

## Library Structure:

NeuralNetwork\
|-- neuralnetwork\
|   |-- \_\_init__.py\
|   |-- neural_network.py\
|-- tests\
|   |-- \_\_init__.py\
|   |-- test_neural_network.py\
|-- examples\
|   |-- example_usage.py\
|-- docs\
|-- setup.py\
|-- README.md\
|-- LICENSE

## Descriptions:

### **NeuralNetwork Object:**

*`NeuralNetwork(array layer_sizes)`*

 #### Properties:
 - NeuralNetwork.weights: 
   > Weights of the neural network ~ Array of `<Numpy array [int]>`
 - NeuralNetwork.biases:
   > Biases of the neural network ~ Array of < Numpy array [int] >
 - NeuralNetwork.layer_sizes
   > An array of the sizes of each layer
 - NeuralNetwork.num_layers
   > Number of layers [int]

 #### Member Functions:
 - NeuralNetwork.sigmoid(numpy array x):
   > Sigmoid Activation Function
 - NeuralNetwork.sigmoid_derivative(numpy array x):
   > Sigmoid Derivative Function
 - NeuralNetwork.forward(numpy array inputs):
   > Returns output of a feedforward across the neural network.
 - NeuralNetwork.backward(numpy array inputs, numpy array targets, float learning_rate):
   > Performs backward propogation though the neural network, updating weights and biases.
 - NeuralNetwork.train_batch(numpy array training_inputs, numpy array training_targets, int epochs, float learning_rate):
   > Performs Batch Gradient Descent to train the network with a batch of data.
 - NeuralNetwork.train_mini_batch(numpy array training_inputs, numpy array training_targets, int batch_size, int epochs, float learning_rate):
   > Performs Batch Gradient Descent to train the network with a mini-batch of data.
 - NeuralNetwork.train_single(numpy array inputs, numpy array targets, float learning rate)
   > Performs Stochastic Gradient Descent to train the network with a single data point.
 - NeuralNetwork.predict(numpy array inputs)
   > Predicts the outputs for a numpy array of inputs.
 - NeuralNetwork.mutate(float mutation_rate)
   > Performs Mutations on the neural network weights and biases.
 - NeuralNetwork.copy(numpy array inputs)
   > Returns a copy of the NeuralNetwork (self).
 
 
 
 
 
