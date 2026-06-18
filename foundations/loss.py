import numpy as np
from numpy.typing import NDArray

class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        total = 0
        coe = -1/len(y_true)
        for i in range(len(y_pred)):
            total += (y_true[i] * math.log(y_pred[i]) + (1-y_true[i]) * math.log(1-y_pred[i]))
        return round(coe*total,4)
    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        loss = 0
        class_len = len(y_true[0])
        coe = -1/len(y_true)
        for i in range(len(y_pred)):
            for j in range(class_len):
                loss += y_true[i][j] * math.log(y_pred[i][j])

        return round(coe*loss,4)
