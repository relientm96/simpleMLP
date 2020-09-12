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

    def activation(self, x):
        # ReLU Activation
        return max(0,x)

    def forward(self, inputs):
        '''
        Applies:
            output = activation(weight * input) + bias
        '''
        compute = np.dot(self.weights.T, inputs)
        output = np.zeros(self.biases.shape)
        for i in range(len(compute)):
            output[i] = self.activation(compute[i]) + self.biases[i]
        return output
''' 
Full Multilayer preceptron
class MLP:
    def __init__(self, ):
        
'''

if __name__ == "__main__":
    layer = Layer(2,3)
    input = np.array([2, 2])
    out   = layer.forward(input)
    pprint(out)