'''
Simple Multi layer Perceptron 
'''
from math import *
import numpy as np

from pprint import *

'''
Defines a single layer in the multi layer preceptron model
Similar to the model.Dense layer of Keras API
'''
class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.weights  = np.random.randn(n_inputs, n_neurons) # Each Weight is initialize gaussianly between (0,1) 
        self.biases   = np.zeros((n_neurons,1))  # Each neuron has it's own unique bias

    def getLayerShape(self):
        # Return the shape of defined layer
        return self.weights.shape

    def activation(self, x):
        # ReLU Activation
        return max(0,x)

    def relu_derivative(self, x):
        # Derivative of ReLu is a Step Function
        return max(0, 1)

    def forward(self, inputs):
        '''
        Applies:
            output = activation(weight * input) + bias
        '''
        # Compute the summation first
        compute = np.dot(self.weights.T, inputs)
        # Initialize an array of outputs, with same shape as bias vector
        output  = np.zeros(self.biases.shape)
        for i in range(len(compute)):
            # Use activation and add bias for each result
            output[i] = self.activation(compute[i]) + self.biases[i]
        return output

'''
Full Multilayer preceptron
'''
class MLP:
    def __init__(self):
        self.layers = []
        self.names  = []

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
        print("===  Defined Architecture ====")
        print("name \t\t layer shape")
        print("---- \t\t ----------")
        for n,l in zip(self.names, self.layers):
            print('{} \t {}'.format(n, l.getLayerShape()))
        print("==============================")

    def forwardRun(self, inputValues):
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
        if (inputValues.shape[0] != self.layers[0].getLayerShape()[0] ):
            # Must conform to same input shape!
            print("First dimension of input {} != first dimension in first input layer, {}"
                    .format(inputValues.shape[0], self.layers[0].getLayerShape()[0]) )
            return None
        else:
            currentInput  = inputValues
            # Perform forward propogation
            for l in self.layers:
                # Compute forward pass for this layer
                output = l.forward(currentInput)
                # Re-assign this output as input to next layer
                currentInput = output
        # Return predictions
        return output.ravel()

    def backPropogation(self)
        '''
        Performs back propogation to adjust weights/biases
        '''
        pass


if __name__ == "__main__":
    mlp = MLP()
    mlp.addLayer(Layer(2,3), "inputLayer")
    mlp.addLayer(Layer(3,2), "outputLayer")
    mlp.printLayers()

    output = mlp.forwardRun(np.array([5,3]))
    print(output)