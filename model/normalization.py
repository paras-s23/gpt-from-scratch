import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        # Scale and shift: out = gamma * x_hat + beta
        # return np.round(your_answer, 5)
        u = (sum(x)/len(x))
        dev = sum((j - u) ** 2 for j in x) / len(x)
        for i in range(len(x)):
            frac = (x[i]-u)/(math.sqrt(dev+1e-5))
            x[i] = frac * gamma[i] + beta[i]
            x[i] = round(x[i],5)
        return x

        