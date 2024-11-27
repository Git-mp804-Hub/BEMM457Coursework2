import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
Power = pd.read_csv('DemandDataPower.csv')
Unemployment = pd.read_csv('Unemployment rate.csv')
Power['Date'] = pd.to_datetime(Power['Date'])
Unemployment['Date'] = pd.to_datetime(Unemployment['Date'], format='%b-%y')
Power.set_index('Date', inplace=True)
Power_monthly = Power.resample('M').mean()
Power_monthly.reset_index(inplace=True)
Unemployment['Date'] = Unemployment['Date'] + pd.offsets.MonthEnd(0)
Unemployment.set_index('Date', inplace=True)
print(Unemployment)
print(Power_monthly)
comparison = pd.merge(Power_monthly, Unemployment, left_on='Date', right_index=True)
comparison.to_csv('UnemploymentTest.csv')
plt.figure(figsize=(10, 6))
plt.scatter(comparison['Seasonally adjusted (%)'], comparison['Demand'], color='purple')
plt.title('Scatter Plot: Power vs Unemployment Rate')
plt.xlabel('Monthly Average Unemployment Rate')
plt.ylabel('Power Demand (Units)')
plt.grid(True)
plt.show()