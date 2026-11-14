import numpy as np

def log_softmax(scores: np.ndarray) -> np.ndarray:
    return scores - np.max(scores) - np.log(sum(np.exp(scores - max(scores))))