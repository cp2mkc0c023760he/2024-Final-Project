import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Loads ticker data from a CSV file.

    Args:
    file_path (str): The path to the CSV file.

    Returns:
    pandas.DataFrame: The loaded data.
    """
    try:
        data = pd.read_csv(file_path, delimiter=',')
        data['Date'] = pd.to_datetime(data['Date'])
        return data
    except Exception as e:
        raise Exception(f"Error loading data: {e}")

def preprocess_data(data):
    """
    Preprocesses the data by splitting it into training and validation sets.

    Args:
    data (pandas.DataFrame): The ticker data.

    Returns:
    tuple: A tuple containing the training and validation data.
    """
    train_data = data.loc[data['Date'].dt.year < 2023, data.columns != 'Date']
    validation_data = data.loc[(data['Date'].dt.year >= 2023) & (data['Date'].dt.year < 2024), data.columns != 'Date']
    return train_data, validation_data

def create_sequences(data, pos, seq_length=60):
    """
    Creates sequences of data points for training the LSTM model.

    Args:
    data (numpy.array): The dataset.
    pos (int): The position of the target variable.
    seq_length (int): The length of the input sequences.

    Returns:
    tuple: A tuple containing the input sequences and corresponding labels.
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, :])
        y.append(data[i, pos])
    return np.array(X), np.array(y)

def market_hours(dataset):
    # Define forex market operational hours (from 5:00pm ET Sunday through 5:00pm ET on Friday)
    market_open_time = pd.Timestamp('17:00:00').time()  # 5:00pm ET
    market_close_time = pd.Timestamp('17:00:00').time()  # 5:00pm ET

    # Create boolean masks to identify rows outside forex market operational hours
    outside_market_hours = (
        (dataset['Date'].dt.dayofweek == 5) |  # Saturday
        ((dataset['Date'].dt.dayofweek == 4) & (dataset['Date'].dt.time >= market_close_time)) |  # Friday after market close
        ((dataset['Date'].dt.dayofweek == 6) & (dataset['Date'].dt.time < market_open_time))  # Sunday before market open
    )

    # Filter dataset to remove rows outside forex market operational hours
    dataset = dataset[~outside_market_hours]
    return dataset


def add_features(dataset):
    # Convert 'Date' column to datetime format
    print("Adding features to the dataset")
    dataset['Date'] = pd.to_datetime(dataset['Date'])

    # Compute volatility measures
    print("Computing volatility measures")
    for ticker in dataset.columns[1:]:
        dataset[f'{ticker}_volatility'] = dataset[ticker].rolling(window=10).std()

    # Compute moving averages
    print("Computing moving averages")
    for ticker in dataset.columns[1:]:
        dataset[f'{ticker}_moving_avg_30'] = dataset[ticker].rolling(window=30).mean()

    # Compute price differentials
    print("Computing price differentials")
    for ticker in dataset.columns[1:]:
        dataset[f'{ticker}_price_diff'] = dataset[ticker].diff()

    # Compute momentum indicators
    print("Computing momentum indicators")
    for ticker in dataset.columns[1:]:
        dataset[f'{ticker}_momentum'] = dataset[ticker].diff(4)
    
    # # Compute relative strength index (RSI)
    # print("Computing relative strength index (RSI)")
    # for ticker in dataset.columns[1:]:
    #     delta = dataset[ticker].diff()
    #     gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    #     loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    #     RS = gain / loss
    #     dataset[f'{ticker}_RSI'] = 100 - (100 / (1 + RS))
    
    # # Compute moving average convergence divergence (MACD)
    # print("Computing moving average convergence divergence (MACD)")
    # for ticker in dataset.columns[1:]:
    #     dataset[f'{ticker}_EMA_12'] = dataset[ticker].ewm(span=12, adjust=False).mean()
    #     dataset[f'{ticker}_EMA_26'] = dataset[ticker].ewm(span=26, adjust=False).mean()
    #     dataset[f'{ticker}_MACD'] = dataset[f'{ticker}_EMA_12'] - dataset[f'{ticker}_EMA_26']
    #     dataset[f'{ticker}_signal_line'] = dataset[f'{ticker}_MACD'].ewm(span=9, adjust=False).mean()  

    # Drop rows with NaN values
    dataset = dataset.dropna()


    return dataset
