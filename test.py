import unittest

import numpy as np

from classes import neural_network
from classes.neuron import Neuron
from classes.neural_network import NeuralNetwork
from classes.layer import Layer


class TestNeuronFunctionality(unittest.TestCase):
    def test_initialisation(self):
        neuron = Neuron(10) # Initialising a neuron with 10 inputs
        self.assertIsInstance(neuron.weights, np.ndarray) # Checking if it's weights are stored in a numpy array, thus random values have been assigned
        self.assertEqual(neuron.weights.shape, (10,)) # Checking that it is in fact a 1D array with 10 values
        self.assertEqual(neuron.weights.dtype, np.float64) # Checking that the values within the array are numpy floats
        self.assertIsInstance(neuron.bias, float)  # Checking that the bias has been initialised as a float

    def test_forward_pass(self):
        neuron = Neuron(5)
        inputs = [5,2,6,3.5,0.6]
        output = neuron.forward_pass(inputs)
        self.assertEqual(neuron.z, np.dot(neuron.weights, inputs) + neuron.bias) # Checking that the forward pass step has been done correctly
        self.assertEqual(neuron.output.dtype, np.float64) # Checking the output is a float
        self.assertNotEqual(neuron.z, neuron.output) # Checking that the activation has been applied


class TestNeuronBackwardPass(unittest.TestCase):
    def setUp(self):
        self.neuron = Neuron(3)
        #Overriding the random values to be able to produce predictable results
        self.neuron.input = np.array([1.0, 2.0, 1.0])
        self.neuron.weights = np.array([0.1, 0.2, 0.5])
        self.neuron.bias = 0.1
        self.neuron.z = 0.9

        def constant_derivative(z):
            return 1.0
        self.neuron.activation_derivative = constant_derivative # Simplifying test so that the activation function returns its input.

    def test_parameter_updates(self):
        derivative_loss = 2.0
        learning_rate = 0.1
        error_term = derivative_loss * 1.0
        expected_dW = self.neuron.input * error_term
        expected_db = error_term
        expected_weights = self.neuron.weights - learning_rate * expected_dW
        expected_bias = self.neuron.bias - learning_rate * expected_db

        self.neuron.backward_pass(derivative_loss, learning_rate)

        np.testing.assert_array_almost_equal(self.neuron.weights, expected_weights) # Almost equal as numpy arithmetic is imprecise due to rounding, sow e use almost equal
        self.assertAlmostEqual(self.neuron.bias, expected_bias) # Again, equal has been used for the same reason as above


class TestLayerFunctionality(unittest.TestCase):
    def setUp(self):
        self.layer = Layer(3, 4)

    def test_forward_pass(self):
        inputs = np.array([1.0, 2.0, 3.0])
        outputs = self.layer.forward_pass(inputs)
        self.assertIsInstance(outputs, np.ndarray)  # Checking that outputs are a numpy array
        self.assertEqual(outputs.shape, (self.layer.num_neurons,))   # Checking that the output shape matches the number of neurons in the layer (1D array)

    def test_backward_pass(self):
        inputs = np.array([1.0, 2.0, 3.0])
        self.layer.forward_pass(inputs) # Running a forward pass

        derivative_loss = np.array([1.0, 0.8, 0.4, 0.1]) # Mock derivative loss
        learning_rate = 0.1
        result = self.layer.backward_pass(derivative_loss, learning_rate)

        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (self.layer.input_size,))


class TestNeuralNetworkFunctionality(unittest.TestCase):
    def setUp(self):
        layers = [3,2,1,5,6,10,100,1,9,1]
        self.neural_network = NeuralNetwork(layers)

    def test_initialization(self):
        self.assertEqual(len(self.neural_network.layers), 9) # There are 9 layer objects as there are only 9 connections of neurons in the network (between first and second, between second and third...)

    def test_invalid_forward_pass(self):
        invalid_x = np.array([1.0, 2.0]) # 2 instead of 3 inputs to the network
        output = self.neural_network.forward_pass(invalid_x)
        self.assertTrue(output == "Invalid Input into the network")

    def test_valid_forward_pass(self):
        valid_x = np.array([1.0, 2.0, 3.0]) # As many inputs as the first layer in the network
        output = self.neural_network.forward_pass(valid_x)
        print(output)
        self.assertIsInstance(output, np.ndarray) # Checking that a correct input produced a numpy array, hence a correct forward pass

    def test_backward_pass(self):
            X = np.array([1.0, 2.0, 3.0])
            Y = np.array([1.5, 1.9, 2.5])
            loss = self.neural_network.backward_pass(X, Y)
            self.assertIsInstance(loss, np.float64) # Checking the output is of the correct type, meaning the los has been calculated and the weights updated.


    def test_training_loop(self):
        x1 = [0.5, 0.3, 0.1]
        x2 = [0.2, 0.9, 0.4]
        x3 = [0.1, 0.2, 0.3]
        x4 = [1, 0.9, 0.8]

        y1 = [1]
        y2 = [0.6]
        y3 = [0.8]
        y4 = [0.1]

        X = np.array([x1, x2, x3, x4])  # Input to the network, in numpy format as I later use the zip function
        Y = np.array([y1, y2, y3, y4])  # The true labels for the network
        epochs = 100
        lr = 0.01
        losses, outputs = self.neural_network.training_loop(X, Y, epochs, lr)

        # Checking the return types
        self.assertIsInstance(losses, list)
        self.assertIsInstance(outputs, np.ndarray)

        self.assertTrue(all(isinstance(loss, np.float64)) for loss in losses) # Checking that every loss is a numpy float (type)
        self.assertEqual(len(losses), epochs) # Checking that the loss has been calculated once every epoch
        self.assertEqual(outputs.shape[0], len(X))  # Checking that we have an output for every input



