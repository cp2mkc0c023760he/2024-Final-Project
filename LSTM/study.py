import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import LSTMModel
from data_preprocessing import load_data, preprocess_data, create_sequences
from util import get_device, train_model, validate_model
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset


def objective(trial):

    # Set the device
    device = get_device()
    print(f"Using device: {device}")

    # Load and preprocess data
    file_path = 'Data/Forex-preprocessed/currencies.csv'  
    dataset = load_data(file_path)
    train_data, validation_data = preprocess_data(dataset)
    dimensions = len(dataset.columns)-1
    pos =dataset.columns.get_loc("EURUSD") -1

    train_data, validation_data= train_data.iloc[-10000:], validation_data.iloc[-1000:]


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



    # Suggested hyperparameters
    num_layers = trial.suggest_int('num_layers', 1, 4)  # Number of LSTM layers
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)  # Learning rate
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])  # Batch sizes
    hidden_dim = trial.suggest_int('hidden_dim', 20, 200)  # Suggest the number of hidden units

    print(f"\nTrial {trial.number}: Testing {num_layers} layers, {hidden_dim} hidden units, lr={lr}, batch size={batch_size}")


    model = LSTMModel(input_dim=dimensions, hidden_dim=hidden_dim, num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)


    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    best_validation_loss = float('inf')
    early_stop_patience = 5
    epochs_no_improve = 0
    num_epochs = 50  

    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)

        validation_loss = validate_model(model, validation_loader, criterion, device)
        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {validation_loss:.4f}")

        # Check for improvement
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve == early_stop_patience:
                print("Early stopping triggered")
                break  # Early stopping

    return best_validation_loss
  

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)  # Number of trials can be adjusted

print("Best trial:")
trial = study.best_trial

print(f"  Number of layers: {trial.params['num_layers']}")
print(f"  Learning rate: {trial.params['lr']}")
print(f"  Batch size: {trial.params['batch_size']}")
print(f"Number of hidden units: {trial.params['hidden_dim']}")


'''
Output:
Trial 0: Testing 1 layers, 195 hidden units, lr=5.912803350552988e-05, batch size=128
Trial 62: Testing 1 layers, 200 hidden units, lr=9.214257869342024e-05, batch size=128
Trial 62 finished with value: 4.047796440254754e-05 and parameters: {'num_layers': 1, 'lr': 9.214257869342024e-05, 'batch_size': 128, 'hidden_dim': 200}
'''