import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

comparison = pd.read_csv('TempTest.csv')
comparison['Date'] = pd.to_datetime(comparison['Date'], format='%Y-%m-%d')
comparison['Temperature^2'] = comparison['MaxTemp'] ** 2
comparison['Weekday'] = (comparison['Date'] == 'Weekday').astype(int) 
print(comparison)
X = comparison[['MaxTemp', 'Temperature^2', 'Weekday']]
X = sm.add_constant(X)  
y = comparison['Demand']
model = sm.OLS(y, X)
results = model.fit()
print(results.summary())
residuals = results.resid
plt.figure(figsize=(12, 6))
plt.plot(comparison['Date'], residuals, label='Residuals', color='purple')
plt.axhline(0, linestyle='--', color='gray', linewidth=0.8)
plt.title('Residual Plot')
plt.xlabel('Date')
plt.ylabel('Residual')
plt.xticks(rotation=45) 
plt.grid(True)
plt.show()

sm.graphics.tsa.plot_acf(residuals, lags=30)
sm.graphics.tsa.plot_pacf(residuals, lags=30)
plt.show()