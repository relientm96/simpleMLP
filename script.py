'''
Training Simple Neural Network
to make predictions for
x^2 + y^2
'''

import numpy as np
import pandas as pd

import pprint as pp

'''
Generating the input training set,
'''

x = np.vstack( (np.random.randint(100, size=100), np.random.randint(100, size=100)) ).T
y = np.array([row[0]**2 + row[1]**2 for row in x]).T

print("=======")
print(x.shape)
print(y.shape)

pp.pprint(x[0:5])

#------------------
# Normalizing Data
#------------------



