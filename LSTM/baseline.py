import pandas as pd
import numpy as np
from data_preprocessing import load_data
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

def calculate_moving_average(data, column_name, window_size):
    """
    Calculate the moving average for a specified column in a pandas DataFrame.

    Parameters:
    - data: pandas DataFrame containing the data.
    - column_name: string, the name of the column to calculate the moving average for.
    - window_size: integer, the number of periods to include in the moving average.

    Returns:
    - moving_average: pandas Series containing the moving averages.
    """
    return data[column_name].rolling(window=window_size).mean()


file_path = 'Data/Forex-preprocessed/currencies.csv'  
dataset = load_data(file_path)

dataset['Date'] = pd.to_datetime(dataset['Date'])  
dataset.set_index('Date', inplace=True)

# Split the dataset into two parts: one for calculating the moving average and one for testing
train_data = dataset.iloc[-20000:-100]  # Use the last 20,000 values excluding the last 100 for training
test_data = dataset.iloc[-100:]  # Keep the last 100 values for testing

# Feature scaling
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(train_data)
validation_set_scaled = sc.transform(test_data)

# Calculate the moving average on the training data
window_size = 200  # The number of sessions for your moving average
moving_avg = calculate_moving_average(train_data, 'EURUSD', window_size)

# Use the last value of the moving average as the prediction for all future values
future_predictions = [moving_avg.iloc[-1]] * 100  # Repeat the last moving average value 100 times

# The actual future values for testing will come from the test_data
actual_future_values = test_data['EURUSD'].values  


# Calculate metrics
mae, mse, rmse, r2 = calculate_metrics(actual_future_values, future_predictions)
print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R-squared: {r2}")


plt.figure(figsize=(14, 7))
plt.plot(dataset['EURUSD'].iloc[-20000:], label='Actual Values')
plt.plot(moving_avg, label='200-Session Moving Average', color='orange')
plt.title('Moving Average vs Actual Values')
plt.legend()
plt.show()

