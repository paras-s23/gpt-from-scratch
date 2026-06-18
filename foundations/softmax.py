import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4
        mv = max(z)
        for i in range(len(z)):
            z[i] = z[i] - mv
        for i in range(len(z)):
            z[i] = math.e**z[i]
        total_sum = sum(z)
        for i in range(len(z)):
            z[i] = round(z[i]/total_sum,4)
        return z


