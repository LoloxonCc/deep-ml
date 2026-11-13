import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    padded = np.pad(input_matrix, padding)

    output_height =(input_height + 2 * padding - kernel_height) // stride + 1
    output_width = (input_width + 2 * padding - kernel_width) // stride + 1

    output_matrix = np.zeros((output_height, output_width))

    for i in range(0, output_height):
        for j in range(0, output_width):
            window = padded[i * stride:i * stride + kernel_height, j * stride:j * stride + kernel_width]
            output_matrix[i][j] = np.sum(window * kernel)
            
    return output_matrix
