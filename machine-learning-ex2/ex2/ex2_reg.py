import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
from plotData import *
import costFunctionReg as cfr
import plotDecisionBoundary as pdb
import predict as predict
import mapFeature as mf


# Load data
# The first two columns contain the exam scores and the third column contains the label.
data = pd.read_csv('ex2data2.txt', sep=',', header=None, names=['Test1', 'Test2', 'Accepted'])

pos = data[data.Accepted.isin([1])]
neg = data[data.Accepted.isin([0])]

fig, ax = plt.subplots()
ax.scatter(pos['Test1'], pos['Test2'], c='b', marker='+', label='y = 1')
ax.scatter(neg['Test1'], neg['Test2'], c='y', marker='o', label='y = 0')
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
plt.legend(loc=1)
# plt.show()

input('Program paused. Press ENTER to continue')

# ===================== Part 1: Regularized Logistic Regression =====================
# In this part, you are given a dataset with data points that are not
# linearly separable. However, you would still like to use logistic
# regression to classify the data points.

# To do so, you introduce more feature to use -- in particular, you add
# polynomial features to our data matrix (similar to polynomial regression)
#

# Add polynomial features

# Note that mapFeature also adds a column of ones for us, so the intercept
# term is handled
cols = data.shape[1]
X = data.iloc[:, :cols - 1]
y = data.iloc[:, cols - 1:]

X = mf.map_feature(X)

# Initialize fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1
lmd = 1

# Compute and display initial cost and gradient for regularized logistic regression
cost, grad = cfr.cost_function_reg(initial_theta, X, y, lmd)

print('Cost at initial theta (zeros): {}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros) - first five values only: \n{}'.format(grad[0:5]))
print('Expected gradients (approx) - first five values only: \n 0.0085\n 0.0188\n 0.0001\n 0.0503\n 0.0115')

input('Program paused. Press ENTER to continue')

# Compute and display cost and gradient with non-zero theta
test_theta = np.ones(X.shape[1])

cost, grad = cfr.cost_function_reg(test_theta, X, y, lmd)

print('Cost at test theta: {}'.format(cost))
print('Expected cost (approx): 2.13')
print('Gradient at test theta - first five values only: \n{}'.format(grad[0:5]))
print('Expected gradients (approx) - first five values only: \n 0.3460\n 0.0851\n 0.1185\n 0.1506\n 0.0159')

input('Program paused. Press ENTER to continue')

# ===================== Part 2: Regularization and Accuracies =====================
# Optional Exercise:
# In this part, you will get to try different values of lambda and
# see how regularization affects the decision boundary
#
# Try the following values of lambda (0, 1, 10, 100).
#
# How does the decision boundary change when you vary lambda? How does
# the training set accuracy vary?
#

# Initializa fitting parameters
initial_theta = np.zeros(X.shape[1])

# Set regularization parameter lambda to 1 (you should vary this)
lmd = 1
# lmd = 0  # 过拟合
# lmd = 100  # 欠拟合

# Optimize
def cost_func(t, X, y):
    t = t.reshape((len(t), 1))
    return cfr.cost_function_reg(t, X, y, lmd)[0]


def grad_func(t, X, y):
    t = t.reshape((len(t), 1))
    return cfr.cost_function_reg(t, X, y, lmd)[1]


res = opt.minimize(fun=cost_func, x0=initial_theta, args=(X, y), method='BFGS', jac=grad_func)
theta = res.x
# Plot boundary
print('Plotting decision boundary ...')
pdb.plot_decision_boundary(theta, X)
plt.title('lambda = {}'.format(lmd))
plt.show()

# Compute accuracy on our training set
p = predict.predict(theta.reshape((len(theta), 1)), X)
count = 0
m = X.shape[0]
for i in range(m):
    if p[i] == y['Accepted'][i]:
        count += 1

print('Train accuracy: {}'.format(count / m * 100))
print('Expected accuracy (with lambda = 1): 83.1 (approx)')

input('ex2_reg Finished. Press ENTER to exit')
