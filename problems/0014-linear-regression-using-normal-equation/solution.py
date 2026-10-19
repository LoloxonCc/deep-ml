import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	X = np.array(X)
	y = np.array(y)
	return np.round(np.linalg.inv(X.T @ X) @ X.T @ y, 4).tolist()