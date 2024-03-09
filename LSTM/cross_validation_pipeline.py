import subprocess

symbols = ['EURAUD', 'EURCAD', 'EURCHF', 'EURCZK', 'EURDKK', 'EURGBP', 'EURHKD', 'EURHUF', 'EURJPY', 'EURMXN', 'EURNOK', 'EURNZD', 'EURPLN', 'EURSEK', 'EURSGD', 'EURTRY', 'EURUSD', 'EURZAR']
for symbol in symbols:
    print(f"Cross Validation for {symbol}")
    subprocess.run(['python', 'main.py', '--option', '3', '--ticker', symbol, '--file_path', 'Data/Forex-preprocessed/currencies.csv'])