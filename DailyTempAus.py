import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from datetime import datetime as dt
from scipy.stats import linregress
Temp = pd.DataFrame(pd.read_csv('tmax.066214.daily.csv'))
Temp = Temp[['Date', 'MaxTemp']]
Temp['Date'] = pd.to_datetime(Temp['Date'], dayfirst=True)
Temp.set_index('Date', inplace=True)
WeeklyTemp = Temp.resample('W')['MaxTemp'].mean()
WeeklyTemp = WeeklyTemp.reset_index()
x = np.arange(len(WeeklyTemp))
y = WeeklyTemp['MaxTemp'].values
slope, intercept, _, _, _ = linregress(x, y)
regression_line = slope * x + intercept
print(slope)
plt.figure(figsize=(10,0))
plt.plot(WeeklyTemp['Date'], WeeklyTemp['MaxTemp'], color='purple')
plt.plot(WeeklyTemp['Date'], regression_line, color='red', linestyle='--', label='Regression Line')
plt.title('Line plot temp over time')
plt.xlabel('Time')
plt.ylabel('Weekly Average Max Temperature (°C)') 
plt.grid(True)
plt.gca().xaxis.set_major_locator(mdates.YearLocator()) 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.show()