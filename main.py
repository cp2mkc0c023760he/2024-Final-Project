import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import time
from util import *
from model import LSTMModel
from data_preprocessing import load_data, preprocess_data, create_sequences
import argparse


#obtained from optuna study
hyperparameters = {
    'num_layers': 1,
    'lr': 9.214257869342024e-05, 
    'batch_size': 128,
    'hidden_dim': 120
}


def train(ticker,file_path,num_epochs=10):
    # Set the device
    device = get_device()
    print(f"Using device: {device}")

    # Load and preprocess data
    dataset = load_data(file_path)
    train_data, validation_data = preprocess_data(dataset)
    dimensions = len(dataset.columns)-1
    pos =dataset.columns.get_loc(ticker) -1

    # Feature scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    training_set_scaled = sc.fit_transform(train_data)
    validation_set_scaled = sc.transform(validation_data)

    # Creating sequences
    X_train, y_train = create_sequences(training_set_scaled, pos)
    X_validation, y_validation = create_sequences(validation_set_scaled, pos)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)
    y_validation_tensor = torch.tensor(y_validation, dtype=torch.float32)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=hyperparameters['batch_size'], shuffle=True)
    validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
    validation_loader = DataLoader(validation_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)



    model = LSTMModel(input_dim=dimensions, hidden_dim=hyperparameters['hidden_dim'],num_layers=hyperparameters['num_layers'])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['lr'])

    model.to(device)


    # Training loop
    training_loss_history = []
    validation_loss_history = []
    for epoch in range(num_epochs):
        start_time = time.time()

        model.train()
        training_loss = train_model(model, train_loader, criterion, optimizer, device)
        training_loss_history.append(training_loss)

        model.eval()
        validation_loss = validate_model(model, validation_loader, criterion, device)
        validation_loss_history.append(validation_loss)
        
        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {training_loss:.4f}, Validation Loss: {validation_loss:.4f}, Time: {epoch_duration:.2f} seconds")

    # Save the model
    model_path = f'new_model_weights_{ticker}.pth'
    torch.save(model.state_dict(), model_path)

    # Plot training and validation loss
    plot_loss(training_loss_history, validation_loss_history)

def predict(ticker,file_path,model_path,num=1000):
    # Set the device 
    device = get_device()
    print(f"Using device: {device}")

    # Load and preprocess data
    dataset = load_data(file_path)
    train_data, validation_data = preprocess_data(dataset)

    # Reduce the size of the validation data to be able to accomodate the window size and get the latest data
    validation_data = validation_data.iloc[-5000:]
    dimensions = len(dataset.columns)-1
    pos =dataset.columns.get_loc(ticker) -1

    # Feature scaling
    sc = MinMaxScaler(feature_range=(0, 1))
    _ = sc.fit_transform(train_data)
    validation_set_scaled = sc.transform(validation_data)
    scaler_target = MinMaxScaler(feature_range=(0, 1))
    scaler_target.fit(train_data.iloc[:, pos].values.reshape(-1, 1))
    X_validation, _ = create_sequences(validation_set_scaled, pos)

    # Convert to PyTorch tensors
    X_validation_tensor = torch.tensor(X_validation[:num], dtype=torch.float32)

   # Create data loaders
    validation_dataset = TensorDataset(X_validation_tensor)
    validation_loader = DataLoader(validation_dataset, batch_size=hyperparameters['batch_size'], shuffle=False)

    # Make predictions
    predicted_prices = make_predictions(
        model_path=model_path,
        data=validation_loader,
        input_dim=dimensions,
        hidden_dim=hyperparameters['hidden_dim'],
        num_layers=hyperparameters['num_layers'],
        device=device,
        scaler=scaler_target
    )

    # Calculate metrics and take the last 100 values
    y_pred = predicted_prices[200:-100]
    y_true = validation_data.iloc[259:, pos]

    # compare the length of the predictions and the actual values and get the minimum
    min_length = min(len(y_pred), len(y_true))
    y_true = y_true[:min_length].to_numpy()

    # print all values and include position
    '''count=0
    for i in range(min_length):
        print(f"{count}- Predicted: {y_pred[i]}, Actual: {y_true[i]}")
        count+=1
        '''

    mae, mse, rmse, r2 = calculate_metrics(y_true, y_pred)
    print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R-squared: {r2}")
    index = np.arange(60, len(y_true) + 60) #take into account the window size
    plot_data(ticker, index, y_true, y_pred)

    # Create comparison arrays using NumPy
    prediction_comparison = create_comparison_array(y_pred)
    validation_comparison = create_comparison_array(y_true)

    # Calculate accuracy using NumPy
    accuracy = calculate_accuracy(prediction_comparison, validation_comparison)
    print(f"Accuracy: {accuracy:.2f}")


    #simulate trading session over 10 days
    total_gains_losses = gain_loss(y_true, y_pred, num_intervals=480)
    print("Total gains/losses over 480 intervals:", total_gains_losses[-1])
 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DL-UPC Project")
    parser.add_argument("--option", default=2, type=int, help="Enter 1 to train the model or 2 to predict")
    parser.add_argument("--ticker", default="EURUSD", help="Enter the ticker")
    parser.add_argument("--file_path", default="Data/Forex-preprocessed/currencies.csv", help="Enter the file path")
    parser.add_argument("--model_path", default="models/LSTM/new_model_weights_EURUSD.pth", help="Enter the model path")

    args = parser.parse_args()

    def handle_menu(input_option, ticker="", file_path="", model_path=""):
        # Create a menu to select the option
        if input_option == 1:
            if ticker == "":
                ticker = "EURUSD"
            if file_path == "":
                file_path = "Data/Forex-preprocessed/currencies.csv"
            train(ticker, file_path)
        elif input_option == 2:
            if ticker == "":
                ticker = "EURUSD"
            if file_path == "":
                file_path = "Data/Forex-preprocessed/currencies.csv"
            if model_path == "":
                model_path = f'models/LSTM/new_model_weights_{ticker}.pth'
            predict(ticker, file_path, model_path)
        else:
            print("Invalid option")

    handle_menu(args.option, args.ticker, args.file_path, args.model_path)
