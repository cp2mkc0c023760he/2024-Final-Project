import torch
import numpy as np

def forecast(model, input_seq):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to track gradients
        # Convert input_seq to a tensor if it's not already
        if not isinstance(input_seq, torch.Tensor):
            input_seq = torch.tensor(input_seq, dtype=torch.float32)
        # Add an extra dimension to input_seq to represent batch size of 1
        input_seq = input_seq.unsqueeze(0)
        # Get the model's prediction
        prediction = model(input_seq)
        return prediction.numpy()  # Convert the prediction to a numpy array
def walk_forward_validation(model, validation_loader, n_input):
    model.eval()  # Ensure the model is in evaluation mode
    predictions = []
    # Iterate over batches in the validation_loader
    for i, (input_seq, target) in enumerate(validation_loader):
        # Predict the next value
        prediction = forecast(model, input_seq)
        # Store the prediction
        predictions.append(prediction)
        # Here you could update your input sequence with the true observation
        # from 'target' if you want to use true values for making the next prediction
    return np.array(predictions)
