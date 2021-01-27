import matplotlib as mpl
mpl.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm
from computeCost import compute_cost
from gradientDescent import gradient_descent

# ===================== Part 1: Plotting =====================
print('Plotting Data...')
# data = np.loadtxt('ex1data1.txt', delimiter=',', usecols=(0, 1))
data = pd.read_csv('ex1data1.txt', sep=',', header=None, names=['Population', 'Profit'])

data.plot(kind='scatter', x='Population', y='Profit', figsize=(12, 8), label='Training Data')
# plt.show()

# plot_data(X, y)

input('Program paused. Press ENTER to continue')

# ===================== Part 2: Gradient descent =====================
print('Running Gradient Descent...')

data.insert(0, 'Ones', 1)
cols = data.shape[1]

X = np.array(data.iloc[:, :cols - 1])
y = np.array(data.iloc[:, cols - 1:])

theta = np.zeros((2, 1))  # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

# Compute and display initial cost
print('Initial cost : ' + str(compute_cost(X, y, theta)) + ' (This value should be about 32.07)')

# theta, J_history = gradient_descent(X, y, theta, alpha, iterations)
theta = np.array([[-3.63029144], [1.16636235]])
print(theta)
print('Theta found by gradient descent: ' + str(theta.reshape(2)))

# Plot the linear fit
line1, = plt.plot(X.iloc[:, 1], np.dot(X, theta), c='r', label='Linear Regression')
plt.legend(loc=2)
# plt.show()

input('Program paused. Press ENTER to continue')

# Predict values for population sizes of 35,000 and 70,000
predict1 = np.dot(np.array([1, 3.5]), theta)
print('For population = 35,000, we predict a profit of {:0.3f} (This value should be about 4519.77)'.format(predict1[0]*10000))
predict2 = np.dot(np.array([1, 7]), theta)
print('For population = 70,000, we predict a profit of {:0.3f} (This value should be about 45342.45)'.format(predict2[0]*10000))

input('Program paused. Press ENTER to continue')

# ===================== Part 3: Visualizing J(theta0, theta1) =====================
print('Visualizing J(theta0, theta1) ...')

theta0_vals = np.linspace(-10, 10, 100)  # 等差数列
theta1_vals = np.linspace(-1, 4, 100)

xs, ys = np.meshgrid(theta0_vals, theta1_vals)  # 获取网格矩阵，用于绘图
J_vals = np.zeros(xs.shape)

# Fill out J_vals
for i in range(0, theta0_vals.size):
    for j in range(0, theta1_vals.size):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        J_vals[i][j] = compute_cost(X, y, t)

J_vals = np.transpose(J_vals)

fig1 = plt.figure(1)
ax = plt.subplot(projection='3d')  # 绘制3D图像
ax.plot_surface(xs, ys, J_vals)  # 绘制曲面图
plt.xlabel(r'$\theta_0$')
plt.ylabel(r'$\theta_1$')

plt.figure(2)
lvls = np.logspace(-2, 3, 30)  # 创建等比数列，控制图密度
plt.contour(xs, ys, J_vals, levels=lvls, norm=LogNorm())  # 绘制等高线图
plt.plot(theta[0], theta[1], c='r', marker="x")
plt.show()
input('ex1 Finished. Press ENTER to exit')
