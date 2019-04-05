"""
Standard Error Calculator

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


def standard_error(preds, labels):
    """
    Description:
         ------
         Calculates the standard error of the predictions vs labels

    Inputs:
         ------
             preds: Predicted values
            labels: Labels


    Returns:
         ------
               se: Standard error
    """
    err = np.subtract(preds, labels)
    sq_err = np.square(err)

    # Variance calculation
    variance = (1 / (preds.shape[0] - 1)) * np.sum(sq_err)

    # Standard deviation calculation
    st_dev = np.sqrt(variance)

    # Standard deviation calculation
    se = st_dev / np.sqrt(preds.shape[0])

    return se


if __name__ == "__main__":

    gen_preds = np.random.normal(5, 2, size=(30, 1))
    gen_labels = np.random.normal(5, 1.5, size=(30, 1))

    SE = standard_error(gen_preds, gen_labels)
    print(SE)

    # Plot results
    plt.plot(gen_labels)
    plt.plot(gen_preds)
    plt.plot(gen_preds - 2 * SE, '--r')
    plt.plot(gen_preds + 2 * SE, '--r')

    plt.show()
