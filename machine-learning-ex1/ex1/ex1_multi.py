import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from featureNormalize import *
from gradientDescent import *
from normalEqn import *

plt.ion()

# ===================== Part 1: Feature Normalization =====================
print('Loading Data...')
# data = np.loadtxt('ex1data2.txt', delimiter=',', dtype=np.int64)
data = pd.read_csv('ex1data2.txt', sep=',', header=None, names=['Size', 'Bedrooms', 'Price'])

cols = data.shape[1]
X1 = data.iloc[:, :cols - 1]
y1 = data.iloc[:, cols - 1:]

# Print out some data points
# print('First 10 examples from the dataset: ')
# for i in range(10):
#     print('x = {}, y = {}'.format(X[i:i+1], y[i:i+1]))
#
# input('Program paused. Press ENTER to continue')

# Scale features and set them to zero mean
print('Normalizing Features ...')

X1, y1, mu, sigma = feature_normalize(X1, y1)

X1.insert(0, 'Ones', 1)  # Add a column of ones to X

# ===================== Part 2: Gradient Descent =====================

# ===================== Your Code Here =====================
# Instructions : We have provided you with the following starter
#                code that runs gradient descent with a particular
#                learning rate (alpha).
#
#                Your task is to first make sure that your functions -
#                computeCost and gradientDescent already work with
#                this starter code and support multiple variables.
#
#                After that, try running gradient descent with
#                different values of alpha and see which one gives
#                you the best result.
#
#                Finally, you should complete the code at the end
#                to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: At prediction, make sure you do the same feature normalization.
#

print('Running gradient descent ...')

# Choose some alpha value
alpha = 0.03
num_iters = 400

# Init theta and Run Gradient Descent
theta = np.zeros((3, 1))
# theta, J_history = gradient_descent(X1, y1, theta, alpha, num_iters)
theta = np.array([[-1.10160108e-16], [8.82183317e-01], [-5.05961547e-02]])
# Plot the convergence graph
# plt.figure()
# plt.plot(np.arange(J_history.size), J_history)
# plt.xlabel('Number of iterations')
# plt.ylabel('Cost J')
# plt.show()
# Display gradient descent's result
print('Theta computed from gradient descent : \n{}'.format(theta))

# Estimate the price of a 1650 sq-ft, 3 br house
# ===================== Your Code Here =====================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.

# ==========================================================
price = np.dot(np.array([1, (1650 - mu[0]) / sigma[0], (3 - mu[1]) / sigma[1]]), theta)[0] * sigma[2] + mu[2]
print('Predicted price of a 1650 sq-ft, 3 br house (using gradient descent) : {:0.3f}'.format(price))

input('Program paused. Press ENTER to continue')

# ===================== Part 3: Normal Equations =====================

print('Solving with normal equations ...')

# ===================== Your Code Here =====================
# Instructions : The following code computes the closed form
#                solution for linear regression using the normal
#                equations. You should complete the code in
#                normalEqn.py
#
#                After doing so, you should complete this code
#                to predict the price of a 1650 sq-ft, 3 br house.
#

# Load data
X2 = data.iloc[:, :cols - 1]
y2 = data.iloc[:, cols - 1:]

# Add intercept term to X
X2.insert(0, 'Ones', 1)

theta = normal_eqn(X2, y2)

# Display normal equation's result
print('Theta computed from the normal equations : \n{}'.format(theta))

# Estimate the price of a 1650 sq-ft, 3 br house
# ===================== Your Code Here =====================
price = np.dot(np.array([1, 1650, 3]), theta)
print(price)
# ==========================================================

print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations) : {:0.3f}'.format(price[0]))

input('ex1_multi Finished. Press ENTER to exit')
