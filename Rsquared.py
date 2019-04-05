"""
R^2 calculator

Rui Nian

sigma = sqrt(sum(xi - mu)^2 / (n - 1))
SE = sigma / sqrt(n)

where:
   sigma: Standard deviation
      xi: i-th sample of x
      mu: Mean of the data
       n: Total amount of examples
      SE: Standard error, follows 68%, 95.5%, 99.7% rule
"""

import numpy as np
import matplotlib.pyplot as plt


def r_squared(preds, labels):
    """
    Description:
         ------
         Calculates the R2 of the predictions vs labels

    Inputs:
         ------
             preds: Predicted values
            labels: Labels


    Returns:
         ------
               r2: Coefficient of determination
    """
    ssr = np.sum(np.square(np.subtract(labels, preds)))
    sst = np.sum(np.square(np.subtract(labels, np.mean(labels))))

    r2 = 1 - np.divide(ssr, sst)

    return r2


if __name__ == "__main__":

    gen_preds = [1, 2, 3, 4, 5]
    gen_labels = [1, 2, 3, 4, 5]

    R2 = r_squared(gen_preds, gen_labels)
    print(R2)
