import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        u = [
            sum(col) / len(x)
            for col in zip(*x)
        ]

        dev = [
            sum((x[i][j] - u[j]) ** 2 for i in range(len(x))) / len(x)
            for j in range(len(x[0]))
        ]

        if training == True:
            x_hat = [
                [
                    (xij - uj) / math.sqrt(devj + eps)
                    for xij, uj, devj in zip(xi, u, dev)
                ]
                for xi in x
            ]

            y = [
                [
                    gamma[j] * x_hat[i][j] + beta[j]
                    for j in range(len(x_hat[0]))
                ]
                for i in range(len(x_hat))
            ]

            running_mean = [
                (1 - momentum) * running_mean[i] + momentum * u[i]
                for i in range(len(u))
            ]

            running_var = [
                (1 - momentum) * running_var[i] + momentum * dev[i]
                for i in range(len(dev))
            ]

            y = [[round(v, 4) for v in row] for row in y]
            running_mean = [round(v, 4) for v in running_mean]
            running_var = [round(v, 4) for v in running_var]

            return y, running_mean, running_var
        else:
            x_hat = [
                [
                    (xij - running_mean[j]) / math.sqrt(running_var[j] + eps)
                    for j, xij in enumerate(xi)
                ]
                for xi in x
            ]

            y = [
                [
                    gamma[j] * x_hat[i][j] + beta[j]
                    for j in range(len(x_hat[0]))
                ]
                for i in range(len(x_hat))
            ]
            y = [[round(v, 4) for v in row] for row in y]
            running_mean = [round(v, 4) for v in running_mean]
            running_var = [round(v, 4) for v in running_var]
            return y, running_mean, running_var