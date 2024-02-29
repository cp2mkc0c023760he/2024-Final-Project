import numpy as np
import torch

import matplotlib.pyplot as plt

from data_preprocessing import load_data, preprocess_data, create_sequences
from predict import make_predictions
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model import LSTMModel
import sklearn.preprocessing



def load_model(model_path, input_dim, hidden_dim,num_layers,device):
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
    model.load_state_dict(torch.load(model_path,map_location=device))
    model.eval()
    return model

def make_predictions(model_path, data, input_dim, hidden_dim, num_layers, device, scaler):
    """
    Make predictions using a trained LSTM model.

    Args:
    model_path (str): Path to the model file.
    data (DataLoader): DataLoader containing the input sequences.
    input_dim (int): Input dimension of the model.
    hidden_dim (int): Hidden dimension of the model.
    num_layers (int): Number of LSTM layers in the model.
    device (torch.device): Device to run the model on.
    scaler (MinMaxScaler): Scaler used to scale the data.
    pos (int): The position of the target variable in the dataset.

    Returns:
    numpy.array: The predicted values.
    """
    model = load_model(model_path, input_dim, hidden_dim, num_layers, device).to(device)
    predictions = []


    for inputs in data:
        inputs = inputs[0].to(device)  # DataLoader returns a tuple (inputs, targets) and you only need inputs
        with torch.no_grad(): 
            outputs = model(inputs).squeeze()  # Get the predictions
        predictions.extend(outputs.cpu().numpy())

    predictions = np.array(predictions)
    predictions_reshaped = predictions.reshape(-1, 1)  # Reshape for compatibility with scaler
    scaled_predictions = scaler.inverse_transform(predictions_reshaped)
    return scaled_predictions


def train_model(model, train_loader, criterion, optimizer, device):
    """
    Train  the LSTM model.

    Args:
    model (LSTMModel): The LSTM model.
    train_loader (DataLoader): DataLoader containing the training data.
    criterion (torch.nn.Module): Loss function.
    optimizer (torch.optim.Optimizer): Optimizer to use for training.
    device (torch.device): Device to run the model on.

    Returns:
    float: The average training loss.
    """
    training_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)  # Squeeze the output

        loss.backward()
        optimizer.step()

        training_loss += loss.item()
    return training_loss / len(train_loader)

def validate_model(model, validation_loader, criterion, device):
    """
    Validate the LSTM model.
    
    Args:
    
    model (LSTMModel): The LSTM model.
    validation_loader (DataLoader): DataLoader containing the validation data.
    criterion (torch.nn.Module): Loss function.
    device (torch.device): Device to run the model on.
    
    Returns:
    float: The average validation loss.
    """
    validation_loss = 0.0
    with torch.no_grad():
        for inputs, targets in validation_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)  # Squeeze the output
            validation_loss += loss.item()
    return validation_loss / len(validation_loader)

def plot_loss(training_loss_history, validation_loss_history):
    plt.figure(figsize=(10, 5))
    plt.plot(training_loss_history, label='Training Loss')
    plt.plot(validation_loss_history, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_data(symbol, index, y_true,y_pred):
    plt.figure(figsize=(12, 6))
    plt.plot(index, y_true, label='Actual Data', color='blue')  
    plt.plot(index, y_pred, label='Predicted Data', color='red')
    plt.title(f'{symbol} Comparison of Predictions and Actual Data')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'Output/images/{symbol}_comparison_new.png')

def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return mae, mse, rmse, r2

def get_device():
    """
    Get the device to run the model on.

    Returns:
    torch.device: The device to run the model on.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Use MPS if available (Apple Silicon)
    elif torch.cuda.is_available():
        return torch.device("cuda")  # Use CUDA if available (NVIDIA GPU)
    else:
        return torch.device("cpu")  # Default to CPU if no GPU is available

def buy_or_sell(close, pred, i):
    """
    Determine whether to buy or sell.

    Args:
    close (numpy.array): Array of closing prices.
    pred (numpy.array): Array of predicted values.
    i (int): Current day index.

    Returns:
    bool: True for Buy, False for Sell.
    """
    if pred[i + 1] > close[i]:
        return True  # Buy
    else:
        return False  # Sell

def gain_loss(close, pred, num_intervals=36):
    """
    Calculate the total gain or loss over a specified number of intervals.

    Args:
    close (numpy.array): Array of closing prices.
    pred (numpy.array): Array of predicted values.
    num_intervals (int): Number of intervals to simulate.

    Returns:
    list: List of cumulative capital over the specified intervals.
    """
    total = []
    capital = 0

    for i in range(num_intervals - 1):  # Loop over the intervals
        diff = 1000 * (close[i + 1] - close[i])  # Calculate the difference in closing prices
        if not buy_or_sell(close, pred, i):
            diff = -diff
        capital += diff
        total.append(capital)

    return total

def create_comparison_array(data):
    # Calculate the difference between consecutive elements
    diffs = np.diff(data, axis=0)
    
    # Use np.sign to get -1, 0, and 1 based on the difference
    comparison_array = np.sign(diffs)
    return comparison_array

def calculate_accuracy(array1, array2):
    # Convert to NumPy arrays and calculate the proportion of matching elements
    array1_np, array2_np = np.array(array1), np.array(array2)
    return np.mean(array1_np == array2_np)