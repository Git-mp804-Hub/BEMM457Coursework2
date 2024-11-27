import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
Power = pd.DataFrame(pd.read_csv('DemandDataPower.csv'))
Power['Date'] = Power['Date'].str.replace(' 00:30:00', '')
Temp = pd.DataFrame(pd.read_csv('tmax.066214.daily.csv'))
Temp = Temp[['Date', 'MaxTemp']]
Temp['Date'] = pd.to_datetime(Temp['Date'], dayfirst=True)
Temp.set_index('Date', inplace=True)
WeeklyTemp = Temp.resample('W')['MaxTemp'].mean()
WeeklyTemp = WeeklyTemp.reset_index()
WeeklyTemp['Date'] = WeeklyTemp['Date'].dt.strftime('%Y-%m-%d')
print(WeeklyTemp)
print(Power)
comparison = pd.merge(WeeklyTemp, Power, on='Date')
comparison.to_csv('TempTest.csv')
plt.figure(figsize=(10,0))
plt.scatter(comparison['MaxTemp'], comparison['Demand'], color='purple')
below_25 = comparison[comparison['MaxTemp'] <= 25]
above_25 = comparison[comparison['MaxTemp'] > 25]
if not below_25.empty:
    coeffs_below = np.polyfit(below_25['MaxTemp'], below_25['Demand'], 1)
    line_below = np.poly1d(coeffs_below)
    slope_below = coeffs_below[0]
    print(slope_below)
    plt.plot(
        below_25['MaxTemp'], 
        line_below(below_25['MaxTemp']), 
        color='blue', 
        label='Regression (x <= 25)'
    )

if not above_25.empty:
    coeffs_above = np.polyfit(above_25['MaxTemp'], above_25['Demand'], 1)
    line_above = np.poly1d(coeffs_above)
    slope_above = coeffs_above[0]
    print(slope_above)
    plt.plot(
        above_25['MaxTemp'], 
        line_above(above_25['MaxTemp']), 
        color='red', 
        label='Regression (x > 25)'
    )

plt.title('Scatter Plot: Power vs Temperature')
plt.xlabel('Weekly Average Max Temperature (°C)')
plt.ylabel('Power Demand (Units)') 
plt.grid(True)
plt.show()