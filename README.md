# 2024-Final-Project
2024-Final-Project


## Download the raw data
This script will download all the public symbols from the company and:
1. Discard the ones that do not have year 2023
2. Merge the different ticks in 10 min slots, doing the median from the given values

To execute it:
``
cd ./Scripts
chmod +x ./*
./Download_all_symbols_slot_10m.sh
``
## Train a forex symbol
1. Create currencies csv file
``cd Data/Forex-preprocessed && cat currencies.csv.gza* > currencies.csv.gz && gunzip currencies.csv.gz``
2. Execute ``python main.py --option 1 --ticker {ticker}``
3. The output will be a model weights

## Inference
1. Execute ``python main.py --option 2 --ticker {ticker} --model_path path``
2. Output a image in Output/images and metrics for stdout 

## Cross Validation + Backtesting
1. Execute ``python main.py --option 3 --ticker {ticker}``
2. Output results for stdout 