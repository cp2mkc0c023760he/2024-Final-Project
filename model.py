import torch.nn as nn



class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        
        self.lstm_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        for i in range(num_layers):
            # For the first layer, input dimension is input_dim
            # For subsequent layers, input dimension is hidden_dim
            layer_input_dim = input_dim if i == 0 else hidden_dim
            self.lstm_layers.append(nn.LSTM(layer_input_dim, hidden_dim, batch_first=True))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
        
        self.dense = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        for lstm, dropout in zip(self.lstm_layers, self.dropout_layers):
            x, _ = lstm(x)
            x = dropout(x)

        # Getting the last output for sequence
        x = x[:, -1, :]
        x = self.dense(x)
        return x

