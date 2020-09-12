# simpleMLP
Simple Multilayer Perceptron built from scratch using numpy.
Currently only uses `ReLu` activation.

## How to use:
1. Install requirements: `pip install -r requirements.txt`
2. Define the Multilayer Perceptron:
```python
mlp = MLP()
mlp.addLayer(Layer(2, 3), "inputLayer")
mlp.addLayer(Layer(3, 1), "outputLayer")
```
3. View Network Architecture:
```python
mlp.printLayers()
# Output:
====== Defined Architecture =======
Layer Name    Layer Shape
------------  -------------
inputLayer    (2, 3)
outputLayer   (3, 1)
==================================
```
4. Define your inputs and output vectors
```python
# Defining data points as 
inputs  = np.array([np.array([2, 2]), np.array([5, 5])])
outputs = np.array([np.array([8]),  np.array([50])])
```
5. Define parameters
```python
lr = 1
iterations = 1
```
5. Train the model
```python
mlp.train(inputs, outputs, iterations, lr)
```
6. View Results after Training!
```python
# Note input_vector must be an np array 
# of similar size as one input for training
mlp.predict( <input_vector> )
```
