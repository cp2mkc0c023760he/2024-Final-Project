import subprocess

symbols = ['EURAUD', 'EURCAD', 'EURCHF', 'EURCZK', 'EURDKK', 'EURGBP', 'EURHKD', 'EURHUF', 'EURJPY', 'EURMXN', 'EURNOK', 'EURNZD', 'EURPLN', 'EURSEK', 'EURSGD', 'EURTRY', 'EURUSD', 'EURZAR']
for symbol in symbols:
    print(f"Predict values for {symbol}")
    model_path = f"models/LSTM/new_model_weights_{symbol}.pth"
    subprocess.run(['python', 'main.py', '--option', '2', '--ticker', symbol, '--file_path', 'Data/Forex-preprocessed/currencies.csv', '--model_path', model_path])