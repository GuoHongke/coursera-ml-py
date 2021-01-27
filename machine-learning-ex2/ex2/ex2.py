import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
from plotData import *
import costFunction as cf
import plotDecisionBoundary as pdb
import predict as predict
from sigmoid import *

# Load data
# The first two columns contain the exam scores and the third column contains the label.
data = pd.read_csv('ex2data1.txt', sep=',', header=None, names=['Exam1', 'Exam2', 'Admitted'])

cols = data.shape[1]
X = data.iloc[:, :cols - 1]
y = data.iloc[:, cols - 1:]

# ===================== Part 1: Plotting =====================
print('Plotting Data with + indicating (y = 1) examples and o indicating (y = 0) examples.')
pos = data[data.Admitted.isin([1])]
neg = data[data.Admitted.isin([0])]

fig, ax = plt.subplots()
ax.scatter(pos['Exam1'], pos['Exam2'], c='b', marker='o', label='Admitted')
ax.scatter(neg['Exam1'], neg['Exam2'], c='r', marker='+', label='Not admitted')
plt.axis([30, 100, 30, 100])
plt.legend(loc=1)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')

input('Program paused. Press ENTER to continue')

# ===================== Part 2: Compute Cost and Gradient =====================
# In this part of the exercise, you will implement the cost and gradient
# for logistic regression. You need to complete the code in
# costFunction.py

# Setup the data array appropriately, and add ones for the intercept term
(m, n) = X.shape

# Add intercept term
X.insert(0, 'Ones', 1)

# Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1))

# Compute and display initial cost and gradient
cost, grad = cf.cost_function(initial_theta, X, y)
np.set_printoptions(formatter={'float': '{: 0.4f}\n'.format})

print('Cost at initial theta (zeros): {:0.3f}'.format(cost))
print('Expected cost (approx): 0.693')
print('Gradient at initial theta (zeros): \n{}'.format(grad))
print('Expected gradients (approx): \n-0.1000\n-12.0092\n-11.2628')

# Compute and display cost and gradient with non-zero theta
test_theta = np.array([[-24], [0.2], [0.2]])
cost, grad = cf.cost_function(test_theta, X, y)

print('Cost at test theta (zeros): {}'.format(cost))
print('Expected cost (approx): 0.218')
print('Gradient at test theta: \n{}'.format(grad))
print('Expected gradients (approx): \n0.043\n2.566\n2.647')

input('Program paused. Press ENTER to continue')

# ===================== Part 3: Optimizing using fmin_bfgs =====================
# In this exercise, you will use a built-in function (opt.fmin_bfgs) to find the
# optimal parameters theta


def cost_func(t, X, y):
    t = t.reshape((len(t), 1))
    return cf.cost_function(t, X, y)[0]


def grad_func(t, X, y):
    t = t.reshape((len(t), 1))
    return cf.cost_function(t, X, y)[1]


initial_theta = np.zeros(3)
# Run fmin_bfgs to obtain the optimal theta
res = opt.minimize(fun=cost_func, x0=initial_theta, args=(X, y), method='BFGS', jac=grad_func)
theta = res.x
cost = res.fun

print('Cost at theta found by fmin: {:0.4f}'.format(cost))
print('Expected cost (approx): 0.203')
print('theta: \n{}'.format(theta))
print('Expected Theta (approx): \n-25.161\n0.206\n0.201')

# Plot boundary
pdb.plot_decision_boundary(theta, X)
plt.show()

input('Program paused. Press ENTER to continue')

# ===================== Part 4: Predict and Accuracies =====================
# After learning the parameters, you'll like to use it to predict the outcomes
# on unseen data. In this part, you will use the logistic regression model
# to predict the probability that a student with score 45 on exam 1 and
# score 85 on exam 2 will be admitted
#
# Furthermore, you will compute the training and test set accuracies of our model.
#
# Your task is to complete the code in predict.py

# Predict probability for a student with score 45 on exam 1
# and score 85 on exam 2

prob = sigmoid(np.array([1, 45, 85]).dot(theta.reshape((len(theta), 1))))
print('For a student with scores 45 and 85, we predict an admission probability of {:0.4f}'.format(prob[0]))
print('Expected value : 0.775 +/- 0.002')

# Compute the accuracy on our training set
p = predict.predict(theta.reshape((len(theta), 1)), X)

count = 0
for i in range(m):
    if p[i] == y['Admitted'][i]:
        count += 1
print('Train accuracy: {}'.format(count / m * 100))
print('Expected accuracy (approx): 89.0')

input('ex2 Finished. Press ENTER to exit')
