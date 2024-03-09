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



