# Multivariable Time Series Forecasting for Trading Abuse# UPC AIDL-2024-Final-Project

This repository is the final project for the UPC School's Artificial Intelligence with Deep Learning postgraduate course (2023-2024). The team members are composed of:

•	Juanjo Nieto (Advisor)
•	Fernando Gordo
•	Inés Badia
•	Oscar Santillana
•	Toni Lopez

## Project Overview

In the world of financial markets, the ability to predict price movements accurately is crucial, especially in mitigating trading abuses. An 'abuser' in our context refers to someone who exploits information or technology to gain an unfair advantage, ultimately profiting at the expense of others. If someone as an abuser develops a deep learning model for price prediction, he gains an unfair advantage in the market, putting trading companies at a disadvantage.
Our goal with this project is to develop robust forecasting models capable of predicting market prices in order to create strategies and measures to mitigate this problem, thereby safeguarding trading companies from such abuses.

## LSTM & Transformers Comparision

LSTM (Long Short-Term Memory)
Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN), offer a powerful framework for modeling sequential data. With their ability to capture long-term dependencies and remember past information, LSTM architectures excel in analyzing time series data such as financial markets. In this project, we explore the efficacy of LSTM networks in forecasting forex prices, harnessing their capacity to learn intricate patterns and dynamics inherent in market behavior.
![image](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/126424332/5e291ada-33d6-4826-9b7c-35435097e5d6)


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
