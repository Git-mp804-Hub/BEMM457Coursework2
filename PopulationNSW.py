import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import time
from datetime import datetime as dt
from scipy.stats import linregress
start_date = '2018-01-01'
end_date = '2023-06-30'
Population = pd.read_csv('PopulationAus.csv', skiprows=range(1, 10))
Population['Date'] = pd.to_datetime(Population['Date'], format='%b-%Y')
Population.set_index('Date', inplace=True) 
Population = Population[(Population.index >= start_date) & (Population.index <= end_date)]
Population = Population.reset_index() 
Population['Date'] = Population['Date'] + pd.offsets.MonthEnd(0) 
x = np.arange(len(Population))
y = Population['Estimated Resident Population ;  Persons ;  New South Wales ;'].values
slope, intercept, _, _, _ = linregress(x, y)
regression_line = slope * x + intercept
print(slope)
plt.figure(figsize=(10,0))
plt.plot(Population['Date'], Population['Estimated Resident Population ;  Persons ;  New South Wales ;'], color='purple')
plt.plot(Population['Date'], regression_line, color='red', linestyle='--', label='Regression Line')
plt.title('Line population over time')
plt.xlabel('Time')
plt.ylabel('Monthly Average Population (Millions, Estimated)') 
plt.grid(True)
plt.gca().xaxis.set_major_locator(mdates.YearLocator()) 
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
plt.xticks(rotation=45)
plt.show()