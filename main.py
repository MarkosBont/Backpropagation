from classes.neural_network import NeuralNetwork
import numpy as np

"""
Network Initialisation
"""
nn = NeuralNetwork([4,6,10,6,4,1]) # Initialises a neural network with 3 layers
nn.network_summary()

"""
Training examples and true values
"""
x1 = [0.5, 0.3, 0.1, 0.7]
x2 = [0.2, 0.9, 0.4, 0.1]
x3 = [0.1, 0.2, 0.3, 0.7]
x4 = [1, 0.9, 0.8, 0.2]

y1 = [1]
y2 = [0.6]
y3 = [0.8]
y4 = [0.1]

X = np.array([x1,x2,x3,x4]) # Input to the network, in this format as we later use the zip function
Y = np.array([y1,y2,y3,y4])


"""
Training Loop
"""
losses, outputs = nn.training_loop(X, Y, epochs=100, lr = 0.1)

print('\n')
print("Outputs:")
for output in outputs:
    print(output)



