import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import linearRegCostFunction as lrcf
import trainLinearReg as tlr
import learningCurve as lc
import polyFeatures as pf
import featureNormalize2 as fn
import plotFit as plotft
import validationCurve as vc


# ===================== Part 1: Loading and Visualizing Data =====================
# We start the exercise by first loading and visualizing the dataset.
# The following code will load the dataset into your environment and pot
# the data.
#

# Load Training data
print('Loading and Visualizing data ...')

# Load from ex5data1:
data = scio.loadmat('ex5data1.mat')
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']

plt.figure()
plt.scatter(X, y, c='r', marker="x")
plt.xlabel('Change in water level (x)')
plt.ylabel('Water folowing out of the dam (y)')
input('Program paused. Press ENTER to continue')

X = np.c_[np.ones(X.shape[0]), X]
Xval = np.c_[np.ones(Xval.shape[0]), Xval]
Xtest = np.c_[np.ones(Xtest.shape[0]), Xtest]

print(X.shape, y.shape, Xval.shape, yval.shape, Xtest.shape, ytest.shape)

# ===================== Part 2: Regularized Linear Regression Cost =====================
# You should now implement the cost function for regularized linear regression
#

theta = np.ones(2)
cost = lrcf.linear_reg_cost_function(theta, X, y, 1)
# cost = lrcf.regularized_cost(theta, X, y, 1)
print('Cost at theta = [1  1]: {:0.6f}\n(this value should be about 303.993192'.format(cost))

input('Program paused. Press ENTER to continue')

# ===================== Part 3: Regularized Linear Regression Gradient =====================
# You should now implement the gradient for regularized linear regression
#

theta = np.ones(2)
grad = lrcf.linear_reg_grad_function(theta, X, y, 1)
print('Gradient at theta = [1  1]: {}\n(this value should be about [-15.303016  598.250744]'.format(grad))

input('Program paused. Press ENTER to continue')

# ===================== Part 4: Train Linear Regression =====================
# Once you have implemented the cost and gradient correctly, the
# train_linear_reg function will use your cost function to train regularzized linear regression.
#
# Write Up Note : The data is non-linear, so this will not give a great fit.
#

# Train linear regression with lambda = 0
lmd = 0

theta = tlr.train_linear_reg(X, y, lmd)
# Plot fit over the data
plt.plot(X[:, 1:], np.dot(X, theta))
# plt.show()
input('Program paused. Press ENTER to continue')

# ===================== Part 5: Learning Curve for Linear Regression =====================
# Next, you should implement the learning_curve function.
#
# Write up note : Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- Figure 3 in ex5.pdf
#

lmd = 0


def plot_learning_curve(X, y, Xval, yval, lmd):
    error_train, error_val = lc.learning_curve(X, y, Xval, yval, lmd)
    plt.figure()
    m = X.shape[0]
    plt.plot(np.arange(1, m+1), error_train, np.arange(1, m+1), error_val)
    plt.title('Learning Curve for Linear Regression lmd = {}'.format(lmd))
    plt.legend(['Train', 'Cross Validation'], loc=1)
    plt.xlabel('Number of Training Examples')
    plt.ylabel('Error')
    plt.axis([0, 13, 0, 150])
    plt.show()


plot_learning_curve(X, y, Xval, yval, lmd)
input('Program paused. Press ENTER to continue')

# ===================== Part 6 : Feature Mapping for Polynomial Regression =====================
# One solution to this is to use polynomial regression. You should now
# complete polyFeatures to map each example into its powers
#

p = 8

# Map X onto Polynomial Features and Normalize
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']

X_poly = pf.poly_features(X, p)
X_poly, mu, sigma = fn.feature_normalize(X_poly)
X_poly = np.c_[np.ones(X.shape[0]), X_poly]

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = pf.poly_features(Xtest, p)
X_poly_test, _, _ = fn.feature_normalize(X_poly_test)
X_poly_test = np.c_[np.ones(X_poly_test.shape[0]), X_poly_test]

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = pf.poly_features(Xval, p)
X_poly_val, mu_val, sigma_val = fn.feature_normalize(X_poly_val)
X_poly_val = np.c_[np.ones(X_poly_val.shape[0]), X_poly_val]

print('Normalized Training Example 1 : \n{}'.format(X_poly[0]))

input('Program paused. Press ENTER to continue')

# ===================== Part 7 : Learning Curve for Polynomial Regression =====================
# Now, you will get to experiment with polynomial regression with multiple
# values of lambda. The code below runs polynomial regression with
# lambda = 0. You should try running the code with different values of
# lambda to see how the fit and learning curve change.
#

lmd = 0.3
theta = tlr.train_linear_reg(X_poly, y, lmd)

# Plot trainint data and fit
plt.figure()
plt.scatter(X, y, c='r', marker="x")
plotft.plot_fit(np.min(X), np.max(X), mu, sigma, theta, p, lmd)

input('Program paused. Press ENTER to continue')


plot_learning_curve(X_poly, y, X_poly_val, yval, lmd)
print('Polynomial Regression (lambda = {})'.format(lmd))

input('Program paused. Press ENTER to continue')

# ===================== Part 8 : Validation for Selecting Lambda =====================
# You will now implement validationCurve to test various values of
# lambda on a validation set. You will then use this to select the
# 'best' lambda value.

lambda_vec, error_train, error_val = vc.validation_curve(X_poly, y, X_poly_val, yval)
plt.figure()
plt.plot(lambda_vec, error_train, lambda_vec, error_val)
plt.legend(['Train', 'Cross Validation'])
plt.xlabel('lambda')
plt.ylabel('Error')
plt.show()
input('Program paused. Press ENTER to continue')


lambda_vec = np.array([0., 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])
for lmd in lambda_vec:
    theta = tlr.train_linear_reg(X_poly, y, lmd)
    print('test cost(lmd={}) = {}'.format(lmd, lrcf.linear_reg_cost_function(theta, X_poly_test, ytest, lmd)))
input('ex5 Finished. Press ENTER to exit')
