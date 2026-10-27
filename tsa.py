# import pandas as pd

# # Load dataset
# df = pd.read_excel("aluminium.xlsx")

# # Preview top rows
# #print(df.head())

# # Convert Date to datetime
# df['Date'] = pd.to_datetime(df['Date'])

# # Set Date as index 
# df = df.set_index('Date')

# # Sort by Date
# df = df.sort_index()

# df = df[['Price']]

# #print(df.head())

# import matplotlib.pyplot as plt

# plt.figure(figsize=(12,5))
# plt.plot(df['Price'], label="Aluminium Price")
# plt.title("Aluminium Price Over Time")
# plt.xlabel("Date")
# plt.ylabel("Price")
# plt.legend()
# plt.show()

# from statsmodels.tsa.stattools import adfuller

# result = adfuller(df['Price'])

# # import matplotlib
# # matplotlib.use('Agg')  # prevents GUI issues

# # import matplotlib.pyplot as plt
# # import pandas as pd

# # df = pd.read_excel("aluminium.xlsx")
# # df['Date'] = pd.to_datetime(df['Date'])
# # df = df.set_index('Date')

# # plt.figure(figsize=(12,5))
# # plt.plot(df['Price'])

# # plt.title("Aluminium Price Over Time")
# # plt.xlabel("Date")
# # plt.ylabel("Price")

# # plt.savefig("aluminium_plot.png")  # saves file instead of GUI window
# # # plt.show()   # if you want, uncomment this after fixing backend

# from statsmodels.tsa.stattools import adfuller

# result = adfuller(df['Price'])

# print("ADF Statistic:", result[0])
# print("p-value:", result[1])
# print("Critical Values:")
# for key, value in result[4].items():
#     print(f'   {key}: {value}')
# 2
# import pandas as pd
# import matplotlib
# matplotlib.use('Agg')   # optional but forces non-GUI backend
# import matplotlib.pyplot as plt
# from statsmodels.tsa.stattools import adfuller

# # Load & prepare
# df = pd.read_excel("aluminium.xlsx")
# df['Date'] = pd.to_datetime(df['Date'])
# df = df.set_index('Date').sort_index()
# df = df[['Price']]

# # Plot and save
# plt.figure(figsize=(12,5))
# plt.plot(df['Price'], label="Aluminium Price")
# plt.title("Aluminium Price Over Time")
# plt.xlabel("Date")
# plt.ylabel("Price")
# plt.legend()
# plt.tight_layout()
# plt.savefig("aluminium_price.png")   # <-- saved image you can open
# plt.close()

# # Run ADF test and print results
# result = adfuller(df['Price'])
# print("ADF Statistic:", result[0])
# print("p-value:", result[1])
# print("Critical Values:")
# for key, value in result[4].items():
#     print(f'   {key}: {value}')

# # tsa_fixed.py -- copy this exactly and run: python tsa_fixed.py
# import matplotlib            # MUST set backend before importing pyplot
# matplotlib.use('Agg')       # non-GUI backend (renders to files)

# import pandas as pd
# import matplotlib.pyplot as plt
# from statsmodels.tsa.stattools import adfuller

# # --- Load & prepare ---
# df = pd.read_excel("aluminium.xlsx")
# df['Date'] = pd.to_datetime(df['Date'])
# df = df.set_index('Date').sort_index()
# # if your column is named something else, change 'Price' accordingly
# df = df[['Price']]

# # --- Plot and save (no GUI) ---
# plt.figure(figsize=(12,5))
# plt.plot(df['Price'], label="Aluminium Price")
# plt.title("Aluminium Price Over Time")
# plt.xlabel("Date")
# plt.ylabel("Price")
# plt.legend()
# plt.tight_layout()
# plt.savefig("aluminium_price.png")   # file written to script folder
# plt.close()                           # free memory, ensure no open window

# # --- Run ADF and print results (will appear immediately in console) ---
# result = adfuller(df['Price'])
# print("ADF Statistic:", result[0])
# print("p-value:", result[1])
# print("Critical Values:")
# for key, value in result[4].items():
#     print(f'   {key}: {value}')

# df['Price_diff'] = df['Price'].diff()
# df_diff = df['Price_diff'].dropna()
# result2 = adfuller(df_diff)
# print("ADF Statistic:", result2[0])
# print("p-value:", result2[1])

# from pmdarima import auto_arima

# model = auto_arima(
#     df['Price'], 
#     seasonal=False, 
#     d=1,           # we determined this from ADF
#     trace=True,    # shows model trials
#     error_action='ignore',
#     suppress_warnings=True
# )

# print(model.summary())

# 4

# tsa_compare.py
import matplotlib            # set backend BEFORE importing pyplot
matplotlib.use('Agg')       # non-GUI backend so script won't hang
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_absolute_error

# --------------------------
# CONFIG
# --------------------------
FNAME = "aluminium.xlsx"       # your Excel file
PRICE_COL = "Price"            # column name for price in the file
TEST_DAYS = 30                 # forecast horizon / test size
PLOT_FILE = "actual_vs_predicted.png"
OUT_CSV = "actual_vs_predicted.csv"

# --------------------------
# LOAD & PREPARE
# --------------------------
df = pd.read_excel(FNAME)
if 'Date' not in df.columns:
    raise KeyError("No 'Date' column found in the file. Rename date column to 'Date'.")
if PRICE_COL not in df.columns:
    raise KeyError(f"No '{PRICE_COL}' column found in the file. Update PRICE_COL variable.")

df['Date'] = pd.to_datetime(df['Date'])
df = df.set_index('Date').sort_index()
df = df[[PRICE_COL]].rename(columns={PRICE_COL: 'Price'})

# drop any rows with missing price
df = df.dropna(subset=['Price'])
if df.shape[0] < TEST_DAYS + 10:
    raise ValueError("Not enough rows in the dataset for the chosen TEST_DAYS. Provide more data.")

# --------------------------
# OPTIONAL: quick stationarity check (prints)
# --------------------------
print("Running ADF on original series (quick check)...")
adf_res = adfuller(df['Price'])
print(f"ADF p-value (original): {adf_res[1]:.6f}")
print(" -> If p-value > 0.05 series is non-stationary (we expect this).")

# --------------------------
# TRAIN / TEST SPLIT
# --------------------------
train = df['Price'][:-TEST_DAYS]
test = df['Price'][-TEST_DAYS:]

print(f"\nData points: total={len(df)}, train={len(train)}, test={len(test)}")

    # --------------------------
# FIT Auto-ARIMA on TRAIN (explicit d=1)
# --------------------------
print("\nFitting auto_arima on training data (this may take a short while)...")
model = auto_arima(
    train,
    seasonal=False,
    d=1,                 # as determined earlier
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

print("\nBest model summary:")
print(model.summary())

# --------------------------
# FORECAST the next TEST_DAYS
# --------------------------
n_periods = TEST_DAYS
forecast = model.predict(n_periods=n_periods)
# ensure forecast array aligns with test index
forecast_index = test.index
forecast = pd.Series(forecast, index=forecast_index, name='Predicted')

# --------------------------
# BUILD COMPARISON DF & METRICS
# --------------------------
comparison_df = pd.DataFrame({
    'Actual': test,
    'Predicted': forecast
})
comparison_df.to_csv(OUT_CSV, index=True)

mae = mean_absolute_error(comparison_df['Actual'], comparison_df['Predicted'])
mape = (np.abs((comparison_df['Actual'] - comparison_df['Predicted']) / comparison_df['Actual']).mean()) * 100

print(f"\nMAE: {mae:.4f}")
print(f"MAPE: {mape:.4f} %")
print(f"\nSaved comparison CSV to: {OUT_CSV}")

# --------------------------
# PLOT and SAVE (no GUI)
# --------------------------
plt.figure(figsize=(12,5))
plt.plot(train.index, train.values, label='Train (history)', linewidth=1)
plt.plot(test.index, test.values, label='Actual (test)', marker='o', linestyle='-', linewidth=1)
plt.plot(forecast.index, forecast.values, label='Predicted', marker='o', linestyle='--', linewidth=1)
plt.title(f"Actual vs Predicted Prices — last {TEST_DAYS} days\nMAE={mae:.4f}, MAPE={mape:.4f}%")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_FILE)
plt.close()
print(f"Saved comparison plot to: {PLOT_FILE}")

# --------------------------
# SHOW TOP ROWS of comparison
# --------------------------
print("\nFirst 10 rows of Actual vs Predicted:")
print(comparison_df.head(10))

# --------------------------
# Done
# --------------------------
print("\nComplete.")
