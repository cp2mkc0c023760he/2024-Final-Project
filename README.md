# Multivariable Time Series Forecasting for Trading Abuse
## UPC AIDL-2024-Final-Project

This repository is the final project for the UPC School's Artificial Intelligence with Deep Learning postgraduate course (2023-2024). The team members are composed of:

* Juanjo Nieto (Advisor)
* Fernando Gordo
* Inés Badia
* Oscar Santillana
* Toni Lopez

## Project Overview

In the world of financial markets, the ability to predict price movements accurately is crucial, especially in mitigating trading abuses. An 'abuser' in our context refers to someone who exploits information or technology to gain an unfair advantage, ultimately profiting at the expense of others. If someone as an abuser develops a deep learning model for price prediction, he gains an unfair advantage in the market, putting trading companies at a disadvantage.
Our goal with is to develop robust forecasting models capable of predicting market prices in order to create strategies and measures to mitigate this problem, thereby safeguarding trading companies from such abuses.

## LSTM & Transformers Comparision

**LSTM**, is a type of recurrent neural network (RNN), offer a powerful framework for modeling sequential data. With their ability to capture long-term dependencies and remember past information, LSTM architectures excel in analyzing time series data such as financial markets. In this project, we explore the efficacy of LSTM networks in forecasting forex prices, harnessing their capacity to learn intricate patterns and dynamics inherent in market behavior.

![image](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/126424332/1a0a88d5-536d-4e8b-9a6b-972625f6ac92)
##### Source: colah's blog

**Transformers**, is a type of neural network architecture that has gained significant attention in recent years, particularly in natural language processing tasks. It utilizes an attention mechanism to weigh the importance of different parts of the input data, allowing it to process sequences of information more effectively. In our context of predicting prices, a Transformer can analyze historical price data and extract relevant patterns and trends, enabling it to make accurate predictions about future price movements. The Transformer's ability to process multiple pieces of information simultaneously makes it particularly well-suited for handling large amounts of sequential data, such as financial time series, which can lead to more accurate and efficient forecasting models.

                    ![image](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/126424332/feeaed92-f661-4548-a846-ca5c381098e2)
##### Source: “Attention is all you need” paper by Vaswani, et al., 2017 [1]

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
