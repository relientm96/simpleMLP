'''
Simple Multi layer Perceptron
'''
from math import *
import numpy as np
from pprint import *
from tabulate import tabulate

'''
Activation Functions & Derivatives,l
Using Relu Activation
'''


def relu(x):
    return max(0, x)


def relu_derivative(x):
    return float(x > 0)


def sigmoid(x):
    return (1.0 / (1 + exp(-x)))


def sigmoid_derivative(x):
    return (1.0)*(1-x)


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
        # Also known as pre-activations
        self.net_sums = np.zeros((n_neurons, 1))
        # The result of applying net_sums with activation for each neuron
        self.activations = np.zeros((n_neurons, 1))
        # Derivatives for this layer's weights
        self.derivatives = np.zeros((n_inputs, n_neurons))

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
            self.activations[i] = sigmoid(self.net_sums[i] + self.biases[i])
        return self.activations


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
        Note: Input shape of layer must match with previous layer
        Args:
            layer - layer you want to add to your sequential model
            name  - name of this specific layer
        '''
        if len(self.layers) > 0:
            # Check that new layer's first dim matches last layer's second dim
            if self.layers[-1].getLayerShape()[1] != layer.getLayerShape()[0]:
                raise Exception("Unmatching Layer Shapes! Please make {} to {}"
                                .format(layer.getLayerShape()[0], self.layers[-1].getLayerShape()[1]))
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

        Args:
            pred - Vector of output predictions
            true - Vector of true values

        Returns:
            Scalar value of loss function
        '''
        return np.average((true - pred)**2)

    def lossFunctionDerivative(self, pred, true):
        '''
        Derivative of the mean square error is just the error
        Args:
            pred - Vector of output predictions
            true - Vector of true values
        Returns:
            Scalar value of loss function
        '''
        return pred - true

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
            raise Exception(
                "Error! No Layers Defined At all to forward propogate")
        if (inputValues.shape[0] != self.layers[0].getLayerShape()[0]):
            # Must conform to same input shape!
            raise Exception("First dimension of input {} != first dimension in first input layer, {}"
                            .format(inputValues.shape[0], self.layers[0].getLayerShape()[0]))
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

    def backProp(self, error, inputs):
        '''
        Given error, back propogate error signal and adjust weights 
        Args:
            error - Error from loss function
            lr    - Learning rate
        '''
        #print("\nBackpropogation: ")
        for i, l in reversed(list(enumerate(self.layers))):

            #print("Layer {} , shape {}".format(i, l.getLayerShape()))
            prev_activations = self.layers[i-1].activations

            # Applying derivative function and multiply with error
            delta = error*sigmoid_derivative(prev_activations)

            # Get new deltas for this layer
            l.derivatives = np.dot(l.activations, delta)

            # Backpropogate the next error
            error = np.dot(delta, l.weights.T)

    def gradientDescent(self, lr):
        '''
        Perform gradient descent using deltas for each layer
        '''
        for l in self.layers:
            print(l.derivatives)

    def train(self, inputs, outputs, epochs, lr):
        '''
        Train the Neural Network
        Args:
            inputs  - List of arrays to be used as inputs
            outputs - List of outputs for each corresponding output
            epochs  - Number of epochs to train 
            lr      - Learning rate when performing gradient descent

            NOTE: shape of input must match output
        '''
        if inputs.shape[0] != outputs.shape[0]:
            raise Exception("Non matching input,output datasets lengths")

        # Start training
        print("\nPerforming Training...")
        for i in range(epochs):
            print("\nIteration {}".format(i))
            sum_error = 0
            for data, target in zip(inputs, outputs):

                # Do forward prop to get prediction
                output = self.forwardProp(data)

                # Error here is the gradient of squared loss function
                error = self.lossFunctionDerivative(output, target)[0]

                # Perform backpropogation to generate deltas
                self.backProp(error, data)

                # Once we have deltas generated, perform gradient descent
                self.gradientDescent(lr)

                # Report error for this epoch
                sum_error += self.lossFunction(output, target)

            print("Epoch {}, loss: {}".format(i, sum_error))

    def predict(self, input):
        '''
        Make a prediction
        '''
        return self.forwardProp(input)


if __name__ == "__main__":

    mlp = MLP()
    mlp.addLayer(Layer(2, 3), "input")
    mlp.addLayer(Layer(3, 1), "output")
    mlp.printLayers()

    # mlp.printNetwork()

    n = 1
    inputs = np.zeros((n, 2))
    outputs = np.zeros((n, 1))
    for i in range(0, n):
        x1 = i+3
        x2 = i+5
        inputs[i] = np.array([x1, x2])
        outputs[i] = np.array([x1**2 + x2**2])
    pprint(inputs)
    pprint(outputs)

    #inputs = np.array([np.array([1, 2])])
    #outputs = np.array([np.array([3])])

    lr = 1
    epochs = 1

    mlp.train(inputs, outputs, epochs, lr)

    # mlp.printNetwork()
    #print("Prediction", mlp.predict(np.array([5, 5])))
