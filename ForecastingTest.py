import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import linregress
import time

start_time = time.time()
def forecast(demand_data, alpha, beta, phi, p, d, q, use_naive, use_ses, use_holt, use_damped, use_arima): 
    naive_forecast, ses_forecast, holt_forecast, damped_forecast, arima_forecast = [], [], [], [], []
    
    for t in range(1, len(demand_data)):
        datatimeseries = demand_data[:t]

        # Naive Forecast
        if use_naive:
            naive_forecast.append(demand_data.iloc[t - 1])
        else:
            naive_forecast.append(np.nan)

        # Simple Exponential Smoothing
        if use_ses and len(datatimeseries) > 1:
            try:
                ses_model = SimpleExpSmoothing(datatimeseries).fit(smoothing_level=alpha)
                ses_forecast.append(ses_model.forecast(1).iloc[0])
            except Exception as e:
                print(f"SES Forecast failed: {e}")
                ses_forecast.append(np.nan)
        else:
            ses_forecast.append(np.nan)

        # Holt's Method
        if use_holt and len(datatimeseries) > 1:
            try:
                holt_model = Holt(datatimeseries).fit(smoothing_level=alpha, smoothing_slope=beta, optimized=False)
                holt_forecast.append(holt_model.forecast(1).iloc[0])
            except Exception as e:
                print(f"Holt's Forecast failed: {e}")
                holt_forecast.append(np.nan)
        else:
            holt_forecast.append(np.nan)

        # Damped Trend Method
        if use_damped and len(datatimeseries) > 1:
            try:
                damped_model = Holt(datatimeseries, damped_trend=True).fit(smoothing_level=alpha, smoothing_slope=beta, damping_trend=phi, optimized=False)
                damped_forecast.append(damped_model.forecast(1).iloc[0])  
            except Exception as e:
                print(f"Damped Trend Forecast failed: {e}")
                damped_forecast.append(np.nan)
        else:
            damped_forecast.append(np.nan)
        
        if use_arima and len(datatimeseries) > max(1, d): 
            best_aic = np.inf
            best_order = None
            best_model = None

            p_values = range(0, 3)
            d_values = range(0, 2)  
            q_values = range(0, 3)

            for ptry in p_values:
                for dtry in d_values:
                    for qtry in q_values:
                        try:
                            temp_model = ARIMA(datatimeseries, order=(ptry, dtry, qtry)).fit()
                            if temp_model.aic < best_aic:
                                best_aic = temp_model.aic
                                best_order = (ptry, dtry, qtry)
                                best_model = temp_model
                                print(best_order)
                        except Exception:
                            continue  
            
            if best_model is not None:
                try:
                    forecast_output = best_model.forecast(steps=1)
                    if isinstance(forecast_output, np.ndarray):
                        arima_forecast.append(forecast_output[0])
                    else:
                        arima_forecast.append(forecast_output)
                except Exception as e:
                    print(f"ARIMA Forecast failed at best order {best_order}: {e}")
                    arima_forecast.append(np.nan)
            else:
                arima_forecast.append(np.nan)
        else:
            arima_forecast.append(np.nan)
    
    forecasts = {
        'Naive': pd.Series([np.nan] + naive_forecast, index=demand_data.index),
        'SES': pd.Series([np.nan] + ses_forecast, index=demand_data.index),
        'Holt': pd.Series([np.nan] + holt_forecast, index=demand_data.index),
        'Damped': pd.Series([np.nan] + damped_forecast, index=demand_data.index),
        'ARIMA': pd.Series([np.nan] + arima_forecast, index=demand_data.index)
    }
    return forecasts

st.title("Demand Forecasting Tool by Michael Perkins")
st.write("Upload a CSV file with 'Date' and 'Demand' columns, preferably in dd/mm/yy format.")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file:
    data = pd.read_csv(uploaded_file, parse_dates=['Date'], dayfirst=True)
    data.set_index('Date', inplace=True)
    demand_data = data['Demand']
    demand_data = demand_data.asfreq('W')
    demand_data.to_csv("DemandDataPower.csv")

    alpha_input = st.text_input("Alpha (smoothing level, 0-1)", "0.2")
    beta_input = st.text_input("Beta (smoothing slope, 0-1)", "0.2")
    phi_input = st.text_input("Phi (damping trend, 0-1)", "0.9")
    p_input = st.text_input("ARIMA p (AR order)", "1")
    d_input = st.text_input("ARIMA d (Differencing order)", "1")
    q_input = st.text_input("ARIMA q (MA order)", "0")

    try:
        alpha = float(alpha_input)
        beta = float(beta_input)
        phi = float(phi_input)
        p = int(p_input)
        d = int(d_input)
        q = int(q_input)
        if not (0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= phi <= 1):
            st.warning("Please enter valid values for alpha, beta, and phi (0-1).")
            alpha, beta, phi = 0.2, 0.2, 0.9
    except ValueError:
        st.warning("Invalid input; using default values of alpha=0.2, beta=0.2, phi=0.9, p=1, d=1, q=0.")
        alpha, beta, phi, p, d, q = 0.2, 0.2, 0.9, 1, 1, 0
    
    use_naive = st.checkbox("Use Naive Forecast", value=True)
    use_ses = st.checkbox("Use Simple Exponential Smoothing", value=True)
    use_holt = st.checkbox("Use Holt's Linear Trend Method", value=True)
    use_damped = st.checkbox("Use Damped Trend Method", value=True)
    use_arima = st.checkbox("Use ARIMA Forecast", value=True)

    forecasts = forecast(demand_data, alpha, beta, phi, p, d, q, use_naive, use_ses, use_holt, use_damped, use_arima)

    plt.figure(figsize=(10, 6))
    plt.plot(demand_data, label='Actual Demand', marker='o')
    for method, forecast in forecasts.items():
        if any(~forecast.isna()): 
            plt.plot(forecast, label=f'{method} Forecast', linestyle='--')
    
    x = np.arange(len(demand_data))
    y = demand_data.values
    slope, intercept, _, _, _ = linregress(x, y)
    regression_line = slope * x + intercept
    plt.plot(demand_data.index, regression_line, label='Trend Line (Regression)', color='red')
    plt.title(f"Demand Forecasting (Trend Slope: {slope:.4f})")
    plt.xlabel("Date")
    plt.ylabel("Demand")
    plt.legend()
    plt.xticks(rotation=45)
    st.pyplot(plt)

    st.write(f"**Trend Slope:** {slope:.4f}")
    if slope > 0:
        st.write("The trend is positive, meaning increasing demand over time.")
    elif slope < 0:
        st.write("The trend is negative, meaning decreasing demand over time.")
    else:
        st.write("The trend is flat, meaning stable demand over time.")
    
    print(" --- %s seconds ---" % (time.time() - start_time))
    forecast_df = pd.DataFrame(forecasts)
    forecast_df.to_csv("forecast_results.csv")
    st.write("Forecast results saved to 'forecast_results.csv'")