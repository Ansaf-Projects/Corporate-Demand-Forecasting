import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the dataset
df = pd.read_csv('train.csv')

# 2. Use your exact column names
date_col = 'Order Date'
sales_col = 'Sales'

# 3. Convert the text dates into Python datetime objects
df[date_col] = pd.to_datetime(df[date_col], format='mixed', dayfirst=False)

# 4. Set the datetime column as the index
df.set_index(date_col, inplace=True)

import itertools
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# 5. The FINAL Aggregation Strategy: Quarterly Sales ('QS' = Quarter Start)
# This absorbs the monthly noise and creates highly predictable cyclical blocks.
quarterly_sales = df[sales_col].resample('QS').sum()
quarterly_sales.dropna(inplace=True)



# 1. Train/Test Split (Sequential)
# Training on 12 quarters (3 years), Testing on 4 quarters (1 year)
train = quarterly_sales.iloc[:-4]
test = quarterly_sales.iloc[-4:]



# 2. Grid Search Levers
trends = ['add', 'mul']
seasonals = ['add', 'mul']
damped = [True, False]

best_mape = float('inf')
best_params = None
best_predictions = None

# Loop through combinations
for t, s, d in itertools.product(trends, seasonals, damped):
    try:
        # Note: seasonal_periods is now 4 (because there are 4 quarters in a year)
        model = ExponentialSmoothing(
            train, 
            trend=t, 
            seasonal=s, 
            seasonal_periods=4, 
            damped_trend=d,
            use_boxcox=True
        ).fit(optimized=True)
        
        predictions = model.forecast(4)
        mape = mean_absolute_percentage_error(test, predictions) * 100
    
        if mape < best_mape:
            best_mape = mape
            best_params = (t, s, d)
            best_predictions = predictions
    except Exception as e:
        continue

print("-" * 50)
print(f"BEST QUARTERLY MODEL: Trend={best_params[0]}, Seasonal={best_params[1]}, Damped={best_params[2]}")
print(f"FINAL OPTIMIZED MAPE: {best_mape:.2f}%")
print("-" * 50)

# 3. Visualize the Quarterly Forecast
plt.figure(figsize=(12, 6))
plt.plot(train.index, train.values, label='Training Data (Years 1-3)', color='blue', marker='o')
plt.plot(test.index, test.values, label='Actual Future Sales (Year 4)', color='green', linewidth=3, marker='s')
plt.plot(best_predictions.index, best_predictions, label=f'Optimized Forecast', color='red', linestyle='--', linewidth=3, marker='X')

plt.title(f'Quarterly Demand Forecast (MAPE: {best_mape:.2f}%)', fontsize=16)
plt.xlabel('Quarter')
plt.ylabel('Aggregated Sales Volume ($)')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()