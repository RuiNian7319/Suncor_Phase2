import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

Data_path = '/Users/ruinian/Documents/Willowglen/data/2019Optimization_Data/' \
            'pump_data.csv'

data = pd.read_csv(Data_path, encoding='utf-8')

for i in range(int(data.shape[0] / 50)):
    data.iloc[i * 50:i * 50 + 7, :] = np.nan

data.dropna(inplace=True)
data.reset_index(inplace=True, drop=True)
