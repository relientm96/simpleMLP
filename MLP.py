'''
Simple Multi layer Perceptron 
'''
from math import *
import numpy as np
from pprint import *
from tabulate import tabulate

'''
Activation Functions & Derivatives, 
Using Relu Activation
'''


def relu(x):
    # ReLU Activation
    return max(0, x)


def relu_derivative(x):
    # Derivative of ReLu is a Step Function
    return float(x > 0)


def sigmoid(x):
    # Sigmoid Activation
    return


'''
Defines a single Dense layer in the multi layer preceptron model
Similar to the model.Dense layer of Keras API
Currently only uses a relu activation function
'''


class Layer:
    def __init__(self, n_inputs, n_neurons):
        # Each Weight is initialize gaussianly between (0,1)
        self.weights = np.random.randn(n_inputs, n_neurons)
        # Each neuron has it's own unique bias
        self.biases = np.zeros((n_neurons, 1))
        # The net summation from all input weights and inputs for each neuron
        self.net_sums = np.zeros((n_neurons, 1))
        # The result of applying net_sums with activation for each neuron
        self.activations = np.zeros((n_neurons, 1))

    def getLayerShape(self):
        # Return the shape of defined layer
        return self.weights.shape

    def forward(self, inputs):
        '''
        Applies:
            output = activation(weight * input) + bias
        '''
        # Compute the summation first
        self.net_sums = np.dot(self.weights.T, inputs)
        for i in range(len(self.net_sums)):
            # Use activation and add bias for each result
            self.activations[i] = relu(self.net_sums[i] + self.biases[i])
        return self.activations

    def backward(self, delta, lr):
        '''
        Applies:
            weights -= lr*(delta)
            bias    -= lr*(delta)
        '''


'''
Full Multilayer preceptron
-> Layers, a list of `layers` objects 
-> Names,  a list of names for each corresponding layer from Layers
'''


class MLP:
    def __init__(self):
        self.layers = []
        self.names = []

    def addLayer(self, layer, name):
        '''
        Add a layer to sequential model with name
        Args:
            layer - layer you want to add to your sequential model
            name  - name of this specific layer
        '''
        self.layers.append(layer)
        self.names.append(name)

    def printLayers(self):
        '''
        Prints out shapes of layers sequentially
        '''
        print("====== Defined Architecture =======")
        table = []
        for n, l in zip(self.names, self.layers):
            table.append([n, l.getLayerShape()])
        print(tabulate(table, headers=["Layer Name", "Layer Shape"]))
        print("==================================")

    def printNetwork(self):
        '''
        Print Network Weights and Biases
        '''
        print("===== Weights n Biases ======")
        for l in self.layers:
            print("Weights: ")
            pprint(l.weights)
            print("Bias: ")
            pprint(l.biases)
            print("===============")

    def lossFunction(self, pred, true):
        '''
        Computing the mean squared error between prediction and actual vectors
        Referenced from:
            https://github.com/ahmedbesbes/Neural-Network-from-scratch/blob/493be2f4015d345fc68d3addd518c2b127e8c648/nn.py#L44

        Args:
            pred - Vector of output predictions 
            true - Vector of true values
        
        Returns:
            Scalar value of loss function
        '''
        n = len(pred)
        cost = (1./(2*n)) * np.sum((true - pred) ** 2)
        return cost

    def forwardProp(self, inputValues):
        '''
        Do forward propogation from first layer to last.
        Note: InputValues must have the shape first dimension as first layer's first dimension
        Args:
            inputValues - input (must have same #rows as first layer)
        Returns:
            Output of a forward propogation pass
            None if non matching input shape or no layers defined
        '''
        if len(self.layers) == 0:
            # No layers! Exit
            print("Error! No Layers Defined At all to forward propogate")
            return None
        if (inputValues.shape[0] != self.layers[0].getLayerShape()[0]):
            # Must conform to same input shape!
            print("First dimension of input {} != first dimension in first input layer, {}"
                  .format(inputValues.shape[0], self.layers[0].getLayerShape()[0]))
            return None
        else:
            currentInput = inputValues
            # Perform forward propogation
            for l in self.layers:
                # Compute forward pass for this layer
                output = l.forward(currentInput)
                # Re-assign this output as input to next layer
                currentInput = output
        # Return predictions
        return output.ravel()

    def backProp(self, error):
        '''
        Given error, back propogate to adjust weights and biases
        Args:
            error - Error from loss function
        '''
        print("\nBackpropogation: ")

        for l, n in zip(reversed(self.layers), reversed(self.names)):
            delta = error*relu_derivative(l.net_sums)
            print(delta)
            # Pass error to layers

    def train(self, inputs, outputs, iterations, lr):
        '''
        Train the Neural Network
        Args:
            inputs  - List of arrays to be used as inputs
            outputs - List of outputs for each corresponding output
            lr      - Learning rate when performing gradient descent
            NOTE: shape of input must match output  
        '''
        if inputs.shape[0] != outputs.shape[0]:
            print("Non matching input,output datasets")
            return None

        # Start training
        print("\nPerforming Training...")
        for i in range(iterations):
            print("\nIteration {}".format(i))
            for data, target in zip(inputs, outputs):
                print("Training for:", data, "=", target)
                # Do forward prop to get prediction
                output = self.forwardProp(data)
                # Error here is the gradient of squared loss function
                error = self.lossFunction(output, target)
                # Now adjust weights using gradient descent
                print("output:{}, true:{}, Error {}".format(
                    output, target, error))
                # Perform backpropogation to fix weights/biases
                self.backProp(error)


if __name__ == "__main__":

    mlp = MLP()
    mlp.addLayer(Layer(2, 1), "input")
    mlp.printLayers()

    inputs = np.array([np.array([2, 2]), np.array([5, 5])])
    outputs = np.array([np.array([8]),  np.array([50])])
    lr = 1
    iterations = 1

    mlp.train(inputs, outputs, iterations, lr)
