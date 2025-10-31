from classes.neuron import Neuron
import numpy as np

class Layer:
    def __init__(self, input_size, num_neurons):
        self.input_size = input_size
        self.neurons = [Neuron(input_size) for i in range(num_neurons)]  # Creates a list of neurons which are in the layer
        self.num_neurons = num_neurons

    def forward_pass(self, inputs):
        """
        Forward pass step
        """
        outputs = np.array([neuron.forward_pass(inputs) for neuron in self.neurons]) # Numpy array such that we can perform matrix multiplication at a later stage
        return outputs.T  #.T flips the rows and columns of the array (transpose)


    def backward_pass(self, derivative_loss,lr):
        """
        Performs backward pass for the whole layer
        derivative_loss: An array of the partial derivatives of the loss w.r.t each neuron's output
        """
        derivative_inputs = np.zeros(self.input_size) # Initialising an array which will hold the total gradient of the loss

        for i, neuron in enumerate(self.neurons):
            derivative_input = neuron.backward_pass(derivative_loss[i], lr)  # Computing how the loss changes w.r.t each neuron's input
            derivative_inputs += derivative_input  # Summing the contribution of all the neurons

        return derivative_inputs