from classes.layer import Layer
import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        """
        Here layers is a list with the number of neurons in each layer. e.g. [3,4,1] has 3 layers
        """
        self.layers = []
        n = len(layers) - 1
        for i in range(n):
            input_size = layers[i]
            output_size = layers[i + 1] # The number of neurons of the next layer (number of connections)
            self.layers.append(Layer(input_size, output_size))  # Creates a layer with the required number of inputs & outputs


    def forward_pass(self, x):
        """
        Computes the forward pass of the neural network.
        """
        x= np.array(x) # Translates this to a 1D array
        if self.layers[0].input_size != len(x):  # Checks if the input is the size the network is expecting
            return "Invalid Input into the network"

        input = x
        for layer in self.layers:
            input = layer.forward_pass(input)  # Computes the output of the input through each layer, producing a prediction
        return input  # Returns the prediction


    def backward_pass(self, x, y, lr = 0.1):
        # Forward pass
        output = self.forward_pass(x)

        # Computing the loss using MSE loss function
        loss = np.mean(0.5 * (y - output) ** 2)

        # Computing the gradient of the loss w.r.t network's output
        gradient_loss = output - y  # The derivative of the MSE loss function

        # Computing the backward pass through all the layers (starting from the last layer)
        for layer in reversed(self.layers):
            gradient_loss = layer.backward_pass(gradient_loss.flatten(), lr)  # Flatten as we are expecting a 1D array

        return loss

    def training_loop(self, X, Y, epochs, lr):
        """
        A full training loop of the neural network.

        X: contains many training inputs x_i which have corresponding true values in Y, y_i.
        Y: contains the true values y_i corresponding to inputs x_i in X.
        epochs: number of epochs to train the neural network
        lr: learning rate
        """

        losses= []
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(X, Y):
                loss = self.backward_pass(x, y, lr)
                total_loss += loss
            avg_loss = total_loss / len(X)
            losses.append(avg_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, loss: {avg_loss:.4f}")

        outputs = np.array([self.forward_pass(x) for x in X])
        return losses, outputs


    def network_summary(self):
        """
        Prints the summary of the neural network, by listing how many neurons each layer contains.
        """
        print("Network Summary:")
        for i, layer in enumerate(self.layers):
            print(f"Layer {i}: {layer.num_neurons} neurons")
        print("\n")

