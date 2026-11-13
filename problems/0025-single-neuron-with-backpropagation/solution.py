import numpy as np

def dsigma_z(sigma_z):
    return sigma_z * (1 - sigma_z)

def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
    
    N = len(features)
    mse_values = np.zeros(epochs)
    for _ in range(epochs):
        sigma_z = 1 / (1 + np.exp(-(features @ initial_weights + initial_bias)))

        mse_values[_] = np.mean((sigma_z - labels) ** 2)

        errors = sigma_z - labels

        dMSE = 2 / N * (features.T @ (errors * dsigma_z(sigma_z)))
        dMSEb = 2 / N * np.sum(errors * dsigma_z(sigma_z))

        initial_weights = initial_weights - learning_rate * dMSE
        initial_bias = initial_bias - learning_rate * dMSEb

    return np.round(initial_weights, 4).tolist(), np.round(initial_bias, 4).tolist(), np.round(mse_values, 4).tolist()