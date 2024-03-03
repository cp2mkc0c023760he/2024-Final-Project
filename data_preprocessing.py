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



