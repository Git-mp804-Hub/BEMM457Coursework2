import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.stats import linregress
Employment = pd.read_csv('AusEmployment.csv')
plt.figure(figsize=(10,0))
x = np.arange(len(Employment))
y = Employment['Employment New South Wales (000)'].values
slope, intercept, _, _, _ = linregress(x, y)
regression_line = slope * x + intercept
print(slope)
plt.plot(Employment['Date'], Employment['Employment New South Wales (000)'], color='purple')
plt.plot(Employment['Date'], regression_line, color='red', linestyle='--', label='Regression Line')
plt.title('Line Employment over time')
plt.xlabel('Month')
plt.ylabel('Month Average Max Employment') 
plt.grid(True)
plt.show()