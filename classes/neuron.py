import numpy as np

class Neuron:
    def __init__(self, input_size):
        """
        Initialises the neuron's weights and biases.
        """
        self.weights = np.random.randn(input_size) # Assigns random weights for each connection in the previous layer, this results in a list of random weights for each connection
        self.bias = np.random.randn()

        self.output = None
        self.input = None
        self.z = None


    def activate(self, x):
        """
        Applies the sigmoid activation function to the input
        """
        return 1/(1+np.exp(-x)) # Sigmoid activation function


    def activation_derivative(self, x):
        """
        Derivative of the sigmoid activation function. (This is the function used)
        """
        s = self.activate(x)
        return s*(1-s)

    def forward_pass(self, inputs):
        """
        Compute the output of the neuron
        """
        self.input = inputs
        self.z = np.dot(inputs, self.weights) + self.bias  # Apply a dot product between all inputs and the weights, and finally add the bias
        self.output = self.activate(self.z) # Apply the activation function

        return self.output

    def backward_pass(self, derivative_loss, learning_rate):
        """
        Computes the gradient, then updates the loss and bias of the neuron.
        derivative_loss is the derivative of the loss function w.r.t the neuron's output.
        """

        d_activation = self.activation_derivative(self.z)
        error_term = derivative_loss * d_activation  # Total derivative of the loss w.r.t z

        dW = self.input * error_term  # Value of derivative of the weight
        db = error_term  # Value of derivative of the bias

        d_input = self.weights * error_term  # The gradient, for the previous layer used for backpropagation

        # Updates the neurons parameters
        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

        return d_input
