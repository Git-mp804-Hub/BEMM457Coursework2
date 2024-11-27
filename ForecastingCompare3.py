import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
start_date = '2018-01-01'
end_date = '2023-06-30'
Population = pd.read_csv('PopulationAus.csv', skiprows=range(1, 10))
Population['Date'] = pd.to_datetime(Population['Date'], format='%b-%Y')
Population.set_index('Date', inplace=True) 
print("Cleaned Population Data:")
print(Population.head())
Population = Population[(Population.index >= start_date) & (Population.index <= end_date)]
Population = Population.reset_index() 
Population['Date'] = Population['Date'] + pd.offsets.MonthEnd(0) 
print(Population.head())
Power = pd.read_csv('DemandDataPower.csv') 
Power['Date'] = pd.to_datetime(Power['Date']) 
Power.set_index('Date', inplace=True)
Power_monthly = Power.resample('M').mean().reset_index()
comparison = pd.merge(Power_monthly, Population, on='Date')
comparison.to_csv('PopulationTest.csv', index=False)
plt.figure(figsize=(10, 6))
plt.scatter(comparison['Estimated Resident Population ;  Persons ;  New South Wales ;'], comparison['Demand'], color='purple')
plt.title('Scatter Plot: Power vs Population')
plt.xlabel('Population (New South Wales)(Millions)')
plt.ylabel('Power Demand (Units)')
plt.grid(True)
plt.show()