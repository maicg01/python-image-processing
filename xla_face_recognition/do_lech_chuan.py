import numpy as np

def calculate_std_dev(matrix):
    # Calculate the column-wise mean
    mean_cols = np.mean(matrix, axis=0)
    # Calculate the squared difference from the mean for each element in each column
    squared_diff = np.square(matrix - mean_cols)
    # Calculate the sum of squared differences
    sum_squared_diff = np.sum(squared_diff, axis=0)
    # Take the square root of the sum of squared differences and divide by the number of elements in each column
    std_dev = np.sqrt(sum_squared_diff / matrix.shape[0])
    
    return std_dev

# Square matrix 3x3
matrix = np.array([[1, 0.3, 0.6],
                   [0.5, 1, 0.9],
                   [0.85, 0.63, 1]])

std_dev = calculate_std_dev(matrix)
print(std_dev)