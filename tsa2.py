# tsa2_full_with_titles.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Load data
# --------------------------
DATA_FILE = "aluminium.xlsx"
data = pd.read_excel(DATA_FILE)

# Ensure Date is datetime and set as index exactly once
data['Date'] = pd.to_datetime(data['Date'])
data = data.set_index('Date').sort_index()

# --------------------------
# Prepare series and handle duplicates
# --------------------------
series = data['Price'].astype(float).copy()

# Detect duplicates and aggregate if present
dup_mask = series.index.duplicated(keep=False)
if dup_mask.any():
    unique_dup_dates = np.unique(series.index[dup_mask])
    print(f"Found duplicate dates: {len(unique_dup_dates)} unique duplicate-date(s) (total duplicate entries: {dup_mask.sum()})")
    for d in unique_dup_dates[:5]:
        print("Example duplicate date:", d, "-> values:", series.loc[d].values if hasattr(series.loc[d], "values") else series.loc[d])
    # Aggregate duplicates by mean (change to .first() or .last() if you prefer)
    series = series.groupby(series.index).mean()
    print("Duplicates aggregated by mean. New length:", len(series))

# If index has no freq, reindex to daily and interpolate
if series.index.freq is None:
    full_idx = pd.date_range(start=series.index.min(), end=series.index.max(), freq='D')
    series = series.reindex(full_idx)
    series = series.interpolate(method='time')
    series.index.name = 'Date'

print("Series index freq:", series.index.freq)
print("Series length:", len(series))

# --------------------------
# Plot raw series
# --------------------------
plt.figure(figsize=(12,5))
plt.plot(series, label="Aluminium Price")
plt.title("Aluminium Price Over Time", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.tight_layout()
plt.show()

# --------------------------
# Seasonal decomposition (if long enough)
# --------------------------
from statsmodels.tsa.seasonal import seasonal_decompose
period = 365
if len(series) >= period * 2:
    result = seasonal_decompose(series, model='multiplicative', period=period, extrapolate_trend='freq')
    result.plot()
    plt.suptitle('Seasonal Decomposition of Aluminium Prices over Time Series', fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    print(f"Series too short for multiplicative seasonal_decompose with period={period}. Skipping decomposition.")

# --------------------------
# Exploratory splits and logs
# --------------------------
values = series.to_numpy()
parts = int(len(values) / 3) if len(values) >= 3 else 1
part_1, part_2, part_3 = values[0:parts], values[parts:(parts*2)], values[(parts*2):]
mean_1, mean_2, mean_3 = np.nanmean(part_1), np.nanmean(part_2), np.nanmean(part_3)
var_1, var_2, var_3 = np.nanvar(part_1), np.nanvar(part_2), np.nanvar(part_3)
print('mean1=%f, mean2=%f, mean3=%f' % (mean_1, mean_2, mean_3))
print('variance1=%f, variance2=%f, variance3=%f' % (var_1, var_2, var_3))

log_values = np.log(series)
parts = int(len(log_values) / 3) if len(log_values) >= 3 else 1
part_1, part_2, part_3 = log_values[0:parts], log_values[parts:(parts*2)], log_values[(parts*2):]
mean_1, mean_2, mean_3 = np.nanmean(part_1), np.nanmean(part_2), np.nanmean(part_3)
var_1, var_2, var_3 = np.nanvar(part_1), np.nanvar(part_2), np.nanvar(part_3)
print('log mean1=%f, mean2=%f, mean3=%f' % (mean_1, mean_2, mean_3))
print('log variance1=%f, variance2=%f, variance3=%f' % (var_1, var_2, var_3))

diff1 = log_values.diff().dropna()
parts = int(len(diff1) / 3) if len(diff1) >= 3 else 1
part_1, part_2, part_3 = diff1[0:parts], diff1[parts:(parts*2)], diff1[(parts*2):]
mean_1, mean_2, mean_3 = np.nanmean(part_1), np.nanmean(part_2), np.nanmean(part_3)
var_1, var_2, var_3 = np.nanvar(part_1), np.nanvar(part_2), np.nanvar(part_3)
print('diff mean1=%f, mean2=%f, mean3=%f' % (mean_1, mean_2, mean_3))
print('diff variance1=%f, variance2=%f, variance3=%f' % (var_1, var_2, var_3))

# --------------------------
# ADF test
# --------------------------
from statsmodels.tsa.stattools import adfuller
adf_res = adfuller(diff1)
print("ADF statistic:", adf_res[0])
print("ADF p-value:", adf_res[1])

# --------------------------
# ACF / PACF
# --------------------------
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
log_diff = log_values.diff().dropna()

plt.figure(figsize=(10,4))
plot_acf(log_diff, lags=60, ax=plt.gca()); plt.title('ACF of log-differenced', fontsize=12)
plt.tight_layout(); plt.show()

plt.figure(figsize=(10,4))
plot_pacf(log_diff, lags=60, ax=plt.gca(), method='ywm'); plt.title('PACF of log-differenced', fontsize=12)
plt.tight_layout(); plt.show()

# --------------------------
# ARIMA on log-values (example)
# --------------------------
from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(log_values, order=(1,1,1))
model_fit = model.fit()
print(model_fit.summary())

forecast_log = model_fit.forecast(steps=30)
forecast = np.exp(forecast_log)
print("30-step ARIMA (log back-transformed) forecast:")
print(forecast)

# Plot original + small forecast
plt.figure(figsize=(12,5))
plt.plot(series, label="Original")
start_date = series.index[-1] + pd.Timedelta(days=1)
forecast_index = pd.date_range(start=start_date, periods=len(forecast), freq='D')
plt.plot(forecast_index, forecast.values, label="Forecast (ARIMA on log then exp)")
plt.title('ARIMA(1,1,1) on Log Series — 30-step Forecast', fontsize=14)
plt.xlabel('Date'); plt.ylabel('Price')
plt.legend(); plt.tight_layout(); plt.show()

# --------------------------
# ARIMA on original scale (long forecast)
# --------------------------
model = ARIMA(series, order=(1, 1, 1))
model_fit = model.fit()
print(model_fit.summary())

n_forecast = 300
pred = model_fit.get_forecast(steps=n_forecast)
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int()

start_date = series.index[-1] + pd.Timedelta(days=1)
future_index = pd.date_range(start=start_date, periods=n_forecast, freq='D')

pred_mean = pd.Series(pred_mean.values, index=future_index, name='predicted_mean')
pred_ci.index = future_index

plt.figure(figsize=(12,5))
plt.plot(series, label='Original')
plt.plot(pred_mean, label='Forecast', color='orange')
plt.fill_between(future_index,
                 pred_ci.iloc[:, 0],
                 pred_ci.iloc[:, 1],
                 color='orange', alpha=0.2)
plt.title('ARIMA(1,1,1) Forecast on Original Scale (300 days)', fontsize=14)
plt.xlabel('Date'); plt.ylabel('Price')
plt.legend(); plt.tight_layout(); plt.show()

# --------------------------
# SARIMA (NO exog) on original scale
# --------------------------
from statsmodels.tsa.statespace.sarimax import SARIMAX

sarima_model = SARIMAX(series,
                       order=(1,1,1),
                       seasonal_order=(1,1,1,7),
                       enforce_stationarity=False,
                       enforce_invertibility=False)
sarima_fit = sarima_model.fit(disp=False)
print(sarima_fit.summary())

n_forecast = 300
pred = sarima_fit.get_forecast(steps=n_forecast)
pred_mean = pred.predicted_mean
pred_ci = pred.conf_int()

start_date = series.index[-1] + pd.Timedelta(days=1)
future_index = pd.date_range(start=start_date, periods=n_forecast, freq='D')

pred_mean = pd.Series(pred_mean.values, index=future_index)
pred_ci.index = future_index

plt.figure(figsize=(12,5))
plt.plot(series, label='Original')
plt.plot(pred_mean, label='Forecast', color='orange')
plt.fill_between(future_index,
                 pred_ci.iloc[:, 0],
                 pred_ci.iloc[:, 1],
                 color='orange', alpha=0.2)
plt.title('SARIMA(1,1,1)x(1,1,1,7) Forecast (no exog)', fontsize=14)
plt.xlabel('Date'); plt.ylabel('Price')
plt.legend(); plt.tight_layout(); plt.show()

# --------------------------
# Residual diagnostics (SARIMA residuals)
# --------------------------
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

resid = sarima_fit.resid.dropna()

sm.graphics.tsa.plot_acf(resid, lags=40); plt.title('ACF of SARIMA Residuals', fontsize=12); plt.tight_layout(); plt.show()
sm.graphics.tsa.plot_acf(resid**2, lags=40); plt.title('ACF of Squared SARIMA Residuals', fontsize=12); plt.tight_layout(); plt.show()
print("Ljung-Box:\n", acorr_ljungbox(resid, lags=[10,20], return_df=True))
print("Jarque-Bera:", sm.stats.jarque_bera(resid))
print("ARCH LM p-value:", het_arch(resid)[1])

# --------------------------
# Fit SARIMA on log-scale (no exog) for ARIMA mean + GARCH variance combo
# --------------------------
df = series.to_frame(name="y")

sarima_log_model = SARIMAX(np.log(df['y']),
                          order=(2,1,2),
                          seasonal_order=(1,1,1,7),
                          enforce_stationarity=False,
                          enforce_invertibility=False)
sarima_log_fit = sarima_log_model.fit(disp=False)
print(sarima_log_fit.summary())

# --------------------------
# GARCH on residuals
# --------------------------
from arch import arch_model
resid = sarima_log_fit.resid.dropna()

garch = arch_model(resid, vol='Garch', p=1, q=1).fit(disp='off')
print(garch.summary())

# --------------------------
# Forecast ARIMA mean (on log-scale) + GARCH variance, back-transform (bias correction)
# --------------------------
h = 300
arima_fc = sarima_log_fit.get_forecast(steps=h)
mu = arima_fc.predicted_mean

# create forecast index (daily)
last_date = df.index[-1]
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=h, freq='D')
mu = pd.Series(mu.values, index=forecast_index)

garch_fc = garch.forecast(horizon=h)
var = garch_fc.variance.values[-1]            # length h
var = pd.Series(var, index=forecast_index)

final_pred = np.exp(mu + 0.5 * var)

plt.figure(figsize=(12,5))
plt.plot(df.index, df['y'], label='Actual')
plt.plot(forecast_index, final_pred, color='red', label='Forecast (ARIMA mean + GARCH var)')
plt.xlim(df.index[0], forecast_index[-1])
plt.title('Forecast Using ARIMA Mean + GARCH Variance (Bias-Corrected)', fontsize=14)
plt.xlabel('Date'); plt.ylabel('Price')
plt.legend(); plt.tight_layout(); plt.show()

# --------------------------
# Train/test backtest and metrics (ARIMA+GARCH on log residuals)
# --------------------------
from sklearn.metrics import mean_squared_error, mean_absolute_error

N = 30
if len(df) <= N + 10:
    print("Series too short for the train/test split with N=", N)
else:
    train = df['y'][:-N]
    test  = df['y'][-N:]

    arima_train = ARIMA(np.log(train), order=(2,1,2)).fit()
    resid_train = arima_train.resid.dropna()
    garch_train = arch_model(resid_train, vol='Garch', p=1, q=1).fit(disp='off')

    h = N
    arima_fc = arima_train.get_forecast(steps=h)
    mu_t = arima_fc.predicted_mean

    garch_fc = garch_train.forecast(horizon=h)
    var_t = garch_fc.variance.values[-1]

    mu_t = pd.Series(mu_t.values, index=test.index)
    var_t = pd.Series(var_t, index=test.index)

    pred_t = np.exp(mu_t + 0.5*var_t)

    mse = mean_squared_error(test.values, pred_t.values)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(test.values, pred_t.values)
    mape = (np.abs((test.values - pred_t.values) / test.values)).mean() * 100

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # analytic 95% CI on original scale (bias-corrected)
    lower_t = np.exp(mu_t - 1.96 * np.sqrt(var_t) + 0.5 * var_t)
    upper_t = np.exp(mu_t + 1.96 * np.sqrt(var_t) + 0.5 * var_t)
    inside = ((test >= lower_t) & (test <= upper_t)).sum()
    coverage = inside / len(test)
    print(f"Analytic 95% interval coverage: {coverage:.2%}")

    plt.figure(figsize=(10,4))
    plt.plot(train.index[-200:], train[-200:], label='Train (last 200)')
    plt.plot(test.index, test, label='Test', color='blue')
    plt.plot(pred_t.index, pred_t, label='Forecast', color='red')
    plt.fill_between(pred_t.index, lower_t, upper_t, color='orange', alpha=0.2)
    plt.title('Train/Test Split with ARIMA + GARCH Forecast', fontsize=14)
    plt.xlabel('Date'); plt.ylabel('Price')
    plt.legend(); plt.tight_layout(); plt.show()

# --------------------------
# Save forecast CSVs for optional Plotly usage
# --------------------------
forecast_out_df = pd.DataFrame({
    'pred': final_pred,
    'lower': np.exp(mu - 1.96 * np.sqrt(var) + 0.5 * var),
    'upper': np.exp(mu + 1.96 * np.sqrt(var) + 0.5 * var)
})
forecast_out_df.to_csv("forecast_daily_table.csv")
df[['y']].to_csv("historical.csv")

print("Saved forecast_daily_table.csv and historical.csv.")
