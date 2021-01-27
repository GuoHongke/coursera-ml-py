import numpy as np
import debugInitializeWeights as diw
import nncostfunction as ncf


def check_nn_gradients(lmd):

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generatesome 'random' test data
    theta1 = diw.debug_initialize_weights(hidden_layer_size, input_layer_size)
    theta2 = diw.debug_initialize_weights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to genete X
    X = diw.debug_initialize_weights(m, input_layer_size - 1)
    y = 1 + np.mod(np.arange(1, m + 1), num_labels)

    # Unroll parameters
    nn_params = np.concatenate([theta1.flatten(), theta2.flatten()])

    numgrad = np.zeros(nn_params.size)
    perturb = np.zeros(nn_params.size)

    e = 1e-4

    for p in range(nn_params.size):
        perturb[p] = e
        loss1 = ncf.nn_cost_function(nn_params - perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lmd)
        loss2 = ncf.nn_cost_function(nn_params + perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lmd)

        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0

    grad = ncf.nn_grad_function(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lmd)
    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)

    print('If your backpropagation implementation is correct,\n'
          'the relative difference will be smaller than 10e-9 (assume epsilon=0.0001).\n'
          'Relative Difference: {}\n'.format(diff))
