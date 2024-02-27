import subprocess

symbols = ['EURAUD', 'EURCAD', 'EURCHF', 'EURCZK', 'EURDKK', 'EURGBP', 'EURHKD', 'EURHUF', 'EURJPY', 'EURMXN', 'EURNOK', 'EURNZD', 'EURPLN', 'EURSEK', 'EURSGD', 'EURTRY', 'EURUSD', 'EURZAR']
for symbol in symbols:
    print(f"Training model for {symbol}")
    model_path = f"{symbol}_model.pth"
    subprocess.run(['python', 'main.py', '--option', '1', '--ticker', symbol, '--file_path', 'currencies.csv', '--model_path', model_path])