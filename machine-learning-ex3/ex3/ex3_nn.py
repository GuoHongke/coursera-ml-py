import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import displayData as dd
from predict_nn import predict

# Setup the parameters you will use for this part of the exercise
input_layer_size = 400  # 20x20 input images of Digits
hidden_layer_size = 25  # 25 hidden layers
num_labels = 10         # 10 labels, from 0 to 9
                        # Note that we have mapped "0" to label 10


# ===================== Part 1: Loading and Visualizing Data =====================
# We start the exercise by first loading and visualizing the dataset.
# You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

data = loadmat('ex3data1.mat')
X = data['X']  # 此处无需转置
y = data['y']
m = y.size

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
selected = X[rand_indices[0:100], :]

# dd.display_data(selected)
# plt.show()
input('Program paused. Press ENTER to continue')

# ===================== Part 2: Loading Parameters =====================
# In this part of the exercise, we load some pre-initiated
# neural network parameters

print('Loading Saved Neural Network Parameters ...')

data = loadmat('ex3weights.mat')
theta1 = data['Theta1']
theta2 = data['Theta2']

# ===================== Part 3: Implement Predict =====================
# After training the neural network, we would like to use it to predict
# the labels. You will now implement the "predict" function to use the
# neural network to predict the labels of the training set. This lets
# you compute the training set accuracy.

print(theta1.shape, theta2.shape)
pred = predict(theta1, theta2, X)
curr_count = 0
for i in range(len(pred)):
    if pred[i] == y[i, 0]:
        curr_count += 1
print(curr_count)

print('Training set accuracy: {}'.format(float(curr_count) / m * 100))

input('Program paused. Press ENTER to continue')

# To give you an idea of the network's output, you can also run
# thru the examples one at a time to see what it is predicting


# Randomly permute examples
rp = np.random.permutation(range(m))

for i in range(m):
    print('Displaying Example image')
    example = X[rp[i]]
    example = example.reshape((1, example.size))
    dd.display_data(example)
    plt.show()
    pred = predict(theta1, theta2, example)
    print('Neural network prediction: {} (digit {})'.format(pred, np.mod(pred, 10)))

    s = input('Paused - press ENTER to continue, q + ENTER to exit: ')
    if s == 'q':
        break
