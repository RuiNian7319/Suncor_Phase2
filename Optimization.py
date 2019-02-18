"""
Suncor optimization code for power and DRA.

Assumes 0 time required to get to outlet, i.e., 0 residence time because it is corrected in measurements.
If this is not the case, the flow rates can be delayed accordingly to build a more accurate model.


"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = '/Users/ruinian/Documents/Willowglen/data/'

raw_data = pd.read_csv(path + 'flow_nn_data.csv')

# Power calculation. (Current * Assumed 1500 voltage * 0.11 dollars per kWh)
raw_data['power_cost'] = np.sum(raw_data.iloc[:, 5:], axis=1) * 1500 * 0.00011

# Assume DRA is 0.1 dollars per ppm for both LP and EP. (Based on estimate of $5M per year since power costs ~17M)
raw_data['dra_cost'] = np.sum(raw_data.iloc[:, 1:5], axis=1) * 0.25

# raw_data = raw_data.values
