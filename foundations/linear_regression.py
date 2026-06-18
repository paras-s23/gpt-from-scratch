import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        coe = 1/len(X)
        res = []

        for i in range(len(X)):
            yhat = 0
            for j in range(len(weights)):
                yhat += X[i][j] * weights[j]
            res.append(round(yhat,5))
        return res
                
                

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        tot = 0
        coe = 1/len(ground_truth)
        for i in range(len(model_prediction)):
            tot += np.sum((model_prediction[i] - ground_truth[i]) ** 2)
        return round(tot*coe,5)
