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
        self.biases   = np.zeros((n_neurons, 1))   # Each neuron has it's own unique bias

    def activation(self, x):
        # Using RELU activation
        return max(0,x)

    def forward(self, inputs):
        '''
        Applies:
            output = activation(weight * input) + bias
        '''
        self.output = np.dot(self.weights.T, inputs)
        for i in range(len(self.output)):
            self.output[i] = self.activation(self.output[i]) + self.biases[i][0]
        return self.output

'''
Full Multilayer preceptron
'''        
class MLP:
    def __init__(self, ):
        



if __name__ == "__main__":
    layer = Layer(2,2)
    input = np.array([2, 2])
    out   = layer.forward(input)