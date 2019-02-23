"""
Normalization class Patch 1.0

Patch notes: -

Date of last edit: February 21st, 2019
Rui Nian

Current issues:

"""

import numpy as np
import pickle


# Min max normalization
class MinMaxNormalization:
    """
    Inputs
       -----
            data:  Input feature vectors from the training data


    Attributes
       -----
         col_min:  The minimum value per feature
         col_max:  The maximum value per feature
     denominator:  col_max - col_min


     Methods
        -----
     init:  Builds the col_min, col_max, and denominator
     call:  Normalizes data based on init attributes
    """

    def __init__(self, data):
        self.col_min = np.min(data, axis=0).reshape(1, data.shape[1])
        self.col_max = np.max(data, axis=0).reshape(1, data.shape[1])
        self.denominator = abs(self.col_max - self.col_min)

        # Fix divide by zero, replace value with 1 because these usually happen for boolean columns
        for index, value in enumerate(self.denominator[0]):
            if value == 0:
                self.denominator[0][index] = 1

    def __call__(self, data):
        return np.divide((data - self.col_min), self.denominator)

    def unnormalize(self, data):

        data = np.multiply(data, self.denominator)
        data = data + self.col_min

        return data
