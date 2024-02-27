import torch
import numpy as np
import pandas as pd1
from sklearn.preprocessing import MinMaxScaler
from model import LSTMModel
from data_preprocessing import create_sequences

def load_model(model_path, input_dim, hidden_dim,num_layers):
    """
    Load the trained LSTM model from a file.

    Args:
    model_path (str): Path to the model file.
    input_dim (int): Input dimension of the model.
    hidden_dim (int): Hidden dimension of the model.

    Returns:
    LSTMModel: The loaded model.
    """
    model = LSTMModel(input_dim=input_dim, hidden_dim=hidden_dim,num_layers=num_layers)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def prepare_input_data(data, seq_length, scaler):
    """
    Prepare the input data for the model.

    Args:
    data (numpy.array): The dataset.
    seq_length (int): The length of the input sequences.
    scaler (MinMaxScaler): The scaler used for data normalization.

    Returns:
    torch.Tensor: The prepared input data as a PyTorch tensor.
    """
    data_scaled = scaler.transform(data.reshape(-1, 1))
    X = []
    for i in range(seq_length, len(data_scaled)):
        X.append(data_scaled[i-seq_length:i, 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return torch.tensor(X, dtype=torch.float32)

def predict(model, input_data):
    """
    Make predictions using the LSTM model.

    Args:
    model (LSTMModel): The trained LSTM model.
    input_data (torch.Tensor): The input data for prediction.

    Returns:
    numpy.array: The predicted values.
    """
    with torch.no_grad():
        predicted = model(input_data).squeeze()  # Get the predictions
        predicted_cpu = predicted.cpu()  # Move the tensor to the CPU
        predicted_numpy = predicted_cpu.numpy()
        return predicted_numpy

def inverse_transform_predictions(predictions, scaler):
    """
    Inverse transform the predictions to the original scale.

    Args:
    predictions (numpy.array): The predicted values.
    scaler (MinMaxScaler): The scaler used for data normalization.

    Returns:
    numpy.array: The inverse-transformed predicted values.
    """
    return scaler.inverse_transform(predictions.reshape(-1, 1))

def make_predictions(model_path, data, seq_length, input_dim, hidden_dim, num_layers, device, scaler):
    """
    High-level function to handle the prediction process.

    Args:
    model_path (str): Path to the trained model file.
    data (numpy.array): The dataset for making predictions.
    seq_length (int): The length of the input sequences.
    input_dim (int): Input dimension of the model.
    hidden_dim (int): Hidden dimension of the model.
    device (torch.device): The device to run the model on.
    scaler (MinMaxScaler): Pre-fitted scaler used for data normalization.

    Returns:
    numpy.array: The predicted values.
    """
    # Use the pre-fitted scaler to transform data
    #data_scaled = scaler.transform(data.reshape(-1, 1))
    
    model = load_model(model_path, input_dim, hidden_dim, num_layers).to(device)
    #input_data = prepare_input_data(data, seq_length,scaler).to(device)
    predictions = predict(model, data)
    return inverse_transform_predictions(predictions, scaler)



