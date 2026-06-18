import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        result = {}
        z1 = np.dot(x,np.array(W1).T) + b1
        a1 = np.array([])
        for z in z1:
            a1 = np.append(a1,max(0,z))
        z2 = np.dot(a1,np.array(W2).T) + b2
        result['loss'] = ((z2 - y_true)**2).item()
        dz2 = (2*(z2-y_true))/len(y_true)
        dW2 = np.outer(dz2,a1)
        result['dW2'] = dW2.tolist()
        result['db2'] = dz2
        da1 = np.dot(dz2,W2)
        dz1 = da1 * (z1 > 0)
        dW1 = np.outer(dz1.T,x)
        result['dW1'] = dW1
        result['db1'] = dz1
        result['loss'] = round(result['loss'], 4)
        result['dW2'] = np.round(dW2, 4).tolist()
        result['db2'] = np.round(dz2, 4).tolist()
        result['dW1'] = np.round(dW1, 4).tolist()
        result['db1'] = np.round(dz1, 4).tolist()
        return result

    
        
