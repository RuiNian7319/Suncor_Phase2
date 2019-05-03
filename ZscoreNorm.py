"""
Z-score normalization class Patch 1.0

Patch notes: -

Rui Nian

Current issues:
"""

import numpy as np
import pickle

import matplotlib.pyplot as plt


# Z-score normalization
class ZScoreNorm:
    """
    Inputs
       -----
            data:  Input feature vectors from the training data


    Attributes
       -----
            mean:  The minimum value per feature
             std:  The maximum value per feature
     denominator:  col_max - col_min


     Methods
        -----
     init:  Builds the col_min, col_max, and denominator
     call:  Normalizes data based on init attributes
    """

    def __init__(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

        # Fix divide by zero, replace value with 1 because these usually happen for boolean columns
        for index, value in enumerate(self.std):
            if value == 0:
                self.std[0][index] = 1

    def __call__(self, data):
        """
        Inputs
            -----
               Data:  Data to be normalized.  In shape [Y | X]

        Returns
            -----
               Data:  Normalized data.  In shape [Y | X]
        """

        de_mean = data - self.mean

        return np.divide(de_mean, self.std)

    def unnormalize(self, data):
        """
        Inputs
            -----
               Data:  Data to be unnormalized.  In shape [Y | X]

        Returns
            -----
               Data:  unormalized data.  In shape [Y | X]
        """

        data = np.multiply(data, self.std)
        data = data + self.mean

        return data

    def unnormalize_y(self, data):
        """
        Description
            -----
                Un-normalizes the labels only.  Label must be the first column when building the normalization object.

        Inputs
            -----
               Data:  Data to be unnormalized.  In shape [Y | X]

        Returns
            -----
               Data:  unormalized data.  In shape [Y | X]
        """

        data = np.multiply(data, self.std[0])
        data = data + self.mean[0]

        return data


if __name__ == "__main__":

    # Generate pseudo-random numbers
    unnorm_numbers = np.random.normal(500, 15, size=[20, 100])

    # Construct normalization parameters
    zscoreNorm = ZScoreNorm(unnorm_numbers)

    # Normalized data
    norm_data = zscoreNorm(unnorm_numbers)

    # Test the normalization call method
    for i in range(norm_data.shape[1]):
        plt.plot(norm_data[:, i])
        print('Column {} mean: {:2f}, std: {:2f}'.format(i,
                                                         np.mean(norm_data[:, i]),
                                                         np.std(norm_data[:, i])))

    plt.show()

    # Test the un-normalize method
    re_unnorm_numbers = zscoreNorm.unnormalize(norm_data)

    for i in range(re_unnorm_numbers.shape[1]):
        diff = np.sum(re_unnorm_numbers[:, i] - unnorm_numbers[:, i])
        print('Column {}, diff: {:2f}'.format(i, diff))

    plt.show()

    # Test the un-normalize y method
    re_unnorm_y = zscoreNorm.unnormalize_y(norm_data[:, 0])

    diff = np.sum(re_unnorm_y - unnorm_numbers[:, 0])
    print('Diff: {:2f}'.format(diff))

    plt.show()
