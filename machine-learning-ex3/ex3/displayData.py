import matplotlib.pyplot as plt
import numpy as np


def display_data(x):
    (m, n) = x.shape

    # Set example_width automatically if not passed in
    example_width = np.round(np.sqrt(n)).astype(int)
    example_height = (n / example_width).astype(int)

    # Compute the number of items to display
    display_width = np.floor(np.sqrt(m)).astype(int)
    display_height = np.ceil(m / display_width).astype(int)

    # Between images padding
    pad = 1

    # Setup blank display
    display_cols = pad + display_width * (example_width + pad)
    display_rows = pad + display_height * (example_height + pad)
    display_array = - np.ones((display_rows, display_cols))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for i in range(1, display_rows, example_height + 1):
        for j in range(1, display_cols, example_width + 1):
            # Copy the patch
            # Get the max value of the patch
            display_array[i:i+example_height, j:j+example_width] = x[curr_ex].reshape((example_height, example_width))
            curr_ex += 1

    # Display image
    plt.imshow(display_array, cmap='gray', extent=[-1, 1, -1, 1])
    plt.axis('off')
