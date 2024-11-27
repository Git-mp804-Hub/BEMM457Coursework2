# This was a first test for using forecasting to a demand set, this is useless now.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
train = pd.read_csv('train.csv', parse_dates=['Date'], index_col='Date')
train_filtered = train[(train['Store'] <= 204) & (train['Store'] >= 200) &(train.index.year == 2014) & (train['Open'] == 1)]
sns.set()
#train.to_csv('ForecastRossData.csv')
#train_filtered.to_csv("trainfiltered.csv")
num_stores = train_filtered['Store'].nunique()
fig, axes = plt.subplots(nrows=(num_stores + 1) // 2, ncols=2, figsize=(15, 20))
axes = axes.flatten()
weekly_means = train_filtered.groupby('Store').resample('W-SUN')['Sales'].mean().reset_index()
store_forecasts = {}

for i, store in enumerate(weekly_means['Store'].unique()):
    store_data = weekly_means[weekly_means['Store'] == store].copy()
    store_data = store_data.set_index('Date')['Sales']
    store_data = store_data.asfreq('W-SUN')

    ses_forecast = []
    holt_forecast = []

    for t in range(len(store_data)):
        data_up_to_t = store_data.iloc[:t+1]

        if len(data_up_to_t) < 2:
            ses_forecast.append(np.nan)
            holt_forecast.append(np.nan)
            continue

        try:
            ses_model = SimpleExpSmoothing(data_up_to_t, initialization_method="estimated").fit(
                smoothing_level=.3, optimized=False
            )
            ses_forecast.append(ses_model.forecast(1).iloc[0])

            holt_model = Holt(data_up_to_t).fit()
            holt_forecast.append(holt_model.forecast(1).iloc[0])

        except (KeyError, ValueError) as e:
            print(f"Error for store {store} at index {t}: {e}")
            ses_forecast.append(np.nan)
            holt_forecast.append(np.nan)
            continue

    ses_forecast_series = pd.Series(ses_forecast, index=store_data.index)
    holt_forecast_series = pd.Series(holt_forecast, index=store_data.index)

    store_forecasts[store] = {
        'SES Forecast': ses_forecast_series,
        'Holt Forecast': holt_forecast_series
    }
    
    ses_forecast_series.to_csv('SimpleExpSmooth.csv')
    holt_forecast_series.to_csv('Holts.csv')
    
    axes[i].plot(store_data, marker="o", label='Weekly Sales')
    axes[i].plot(ses_forecast_series, label='SES Forecast', linestyle='--')
    axes[i].plot(holt_forecast_series, label='Holt Forecast', linestyle='--')
    axes[i].set_title(f'Store {store} - Weekly Sales Forecast')
    axes[i].set_ylabel('Sales')
    axes[i].set_xlabel('Date')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].legend()


for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout(pad=3)
plt.show()
