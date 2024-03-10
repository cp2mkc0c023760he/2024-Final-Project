# Multivariable Time Series Forecasting for Trading Abuse

## UPC AIDL-2024-Final-Project

This repository is the final project for the UPC School's Artificial Intelligence with Deep Learning postgraduate course (2023-2024). The team members are composed of:

* Juanjo Nieto (Advisor)
* Fernando G.
* Inés Badia
* Oscar Santillana
* Toni Lopez

## Project Overview

In the world of financial markets, the ability to predict price movements accurately is crucial, especially in mitigating trading abuses. An 'abuser' in our context refers to someone who exploits information or technology to gain an unfair advantage, ultimately profiting at the expense of others. If someone as an abuser develops a deep learning model for price prediction, he gains an unfair advantage in the market, putting trading companies at a disadvantage.

Our goal with is to develop robust forecasting models capable of predicting market prices in order to create strategies and measures to mitigate this problem, thereby safeguarding trading companies from such abuses.

## Custom LSTM VS Transformer-Informer Comparision

**LSTM**, is a type of recurrent neural network (RNN), offer a powerful framework for modeling sequential data. With their ability to capture long-term dependencies and remember past information, LSTM architectures excel in analyzing time series data such as financial markets. In this project, we explore the efficacy of LSTM networks in forecasting forex prices, harnessing their capacity to learn intricate patterns and dynamics inherent in market behavior.

![image](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/126424332/1a0a88d5-536d-4e8b-9a6b-972625f6ac92)
Source: colah's blog

Advantages:

* Good at capturing sequential dependencies in time series data.
* Able to learn features autonomously
* Provides some interpretability.

Disadvantages:

* Has difficulty long-distance dependencies in noisy markets.
* Computationally expensive to train.
* Susceptible to overfitting.

**Transformers**, is a type of neural network architecture that has gained significant attention in recent years, particularly in natural language processing tasks. It utilizes an attention mechanism to weigh the importance of different parts of the input data, allowing it to process sequences of information more effectively. In our context of predicting prices, a Transformer can analyze historical price data and extract relevant patterns and trends, enabling it to make accurate predictions about future price movements. The Transformer's ability to process multiple pieces of information simultaneously makes it particularly well-suited for handling large amounts of sequential data, such as financial time series, which can lead to more accurate and efficient forecasting models.

![image](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/126424332/feeaed92-f661-4548-a846-ca5c381098e2)

Source: “Attention is all you need” paper by Vaswani, et al., 2017 <https://arxiv.org/abs/1706.03762>

Advantages:

* Leverage parallel processing for faster training.
* Better at capturing global dependencies in data.
* Highly scalable.

Disadvantages:

* More complex architecture.
* Require larger amounts of training data.
* Resource-intensive during training.

**Transformer-Informer**, <https://huggingface.co/blog/informer>

An informer is a time series forecasting model that implements enhancements to the basic Transformer, such as ProbSparse attention, to improve performance in predicting probabilistic distributions in multivariate time series.

## Metrics considered

In this project, we have chosen several metrics to assess the performance of our models:

* Mean Absolute Error (MAE): This metric quantifies the average magnitude of errors in the model's predictions. A lower value indicates better accuracy.
* Mean Squared Error (MSE): MSE calculates the average of the squares of the errors between the model's predictions and the actual values. The lower the MSE, the better the model's performance.
* Root Mean Squared Error (RMSE): RMSE is simply the square root of MSE and provides a measure of the spread of errors. Like MSE, this value is sought to be minimized.
* Coefficient of Determination (R-squared): This metric indicates the proportion of variance in the dependent variable that is predictable from the independent variable(s). A value close to 1 indicates a good fit of the model to the data.
* Accuracy: Metric used in classification problems in deep learning. Measures how accurately the model's predictions match the true responses, in our case the trend of the market.

## Computational resources

* Runpod Cloud - Custom LSTM
  * System equiped with GPU RTX 4090

* Colab Pro - Transformer-Informer
  * Option GPU NVIDIA T4 Tensor

## Milestones

### M0. Research

Preliminary research on LSTM and Transformer architectures for time series forecasting.

### M1. Custom dataset

Creating a custom dataset tailored to the particular requirements from our project, including collecting public data and preprocessing historical price data for EUR pairs in ticks each 10 mins

### M2. Selection of symbols to predict (EUR pairs)

Choosing the specific currency pairs (EUR pairs) that will be the focus of the price prediction models.

### M3. Initial versions of LSTM & Transformer

Adapting the base of our Transfomer arquitecture since <https://huggingface.co/blog/informer> and in the LSTM case from pytorch.org.

### M4. Hyper parameter tuning

Fine-tuning the hyperparameters of the LSTM and Transformer models to optimize their performance and accuracy

### M5. Final version and backtesting of the models

Refinement both models and conducting backtesting to assess their performance and validate their effectiveness.

### M6. Results comparison between models

Comparing the results obtained of both models to determine which architecture performs better for the task of price prediction.

### M7. Evaluate different symbols

For assessing their correlation and determine which one is the most predictable.

## Run the models

### LSTM

#### Train a forex symbol

1. Create currencies csv file
``cd Data/ && cat currencies.csv.gza* > currencies.csv.gz && gunzip currencies.csv.gz``
1. Execute ``python main.py --option 1 --ticker {ticker}``
2. The output will be a model weights

#### Inference

1. Execute ``python main.py --option 2 --ticker {ticker} --model_path path``
2. Output a image in Output/images and metrics for stdout

#### Cross Validation + Backtesting

1. Execute ``python main.py --option 3 --ticker {ticker}``
2. Output results for stdout

### Transformer - Informer

The code done does the train, inference, coss validation and generate all the stadistics only by running the collab.
We recomend the use of a T4 system and it takes around 3h to finish.

## LSTM Results  

![311539719-04faa086-06b1-466e-a77a-b4111dd07013_2](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/126424332/8ba22b93-9cf7-4bad-9be9-4bb0aec16b79)

![image](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/126424332/80d1dc5e-58b1-47ff-be5a-3a68fc19355a)

![image](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/126424332/7e8c3f46-fc0a-4d8b-8050-38da2a3541a5)

![image](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/126424332/6f3522c1-e93e-4093-8571-4fe2d77ab9e2)

## Transformer Informer Results

![image](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/126424332/3153cff3-cfce-4717-9c49-4e2cdcc27d4d)

![image](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/126424332/79670264-600a-4970-b10d-45e4e1d07261)

## Results - Comparative

![image](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/126424332/49285541-9c12-430d-926c-aafb10c12e2d)


## Conclusions

* Gained experience applying LSTM/Transformer models for Time Series Forecasting
* Backtesting, pipeline alignment and custom metrics help to compare model results
* For the symbols chosen our LSTM model performs better than the Transformer:
  * They could predict a few symbols pairs (5/18): EUR ( AUD|NOK|NZD|HKD|USD )
    * 50 % < test accuracy (direction) < 76 %
    * 50 % < cross validation accuracy (direction)
* Our models:
  * Seem to detect different patterns (different best predicted symbols)
  * Require more testing on other symbols to evaluate its performance

* Creating a dataset is a complex task (missing data, holidays, timezones, etc..) - we spent maybe to much time trying to create it
* LSTM performs better with only one hidden layer → vanishing gradient
* Missing feature engineering step - since financial data has a lot of indicators we might leverage it. 
* Classical ML algorithms might perform better → initial results of +60% accuracy in test with XGboost and little feature engineering → require less GPU resources (cheaper to use)
* Simple operations (e.g. backtesting) become complex and need to be optimized when dealing with huge dataset.
* Working with git earlier to showcase progress and standardize steps


## References

* "Understanding LSTM Networks" blog post by Christopher Olah (Posted on August 27, 2015) <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>
* "Attention is all you need" paper by Vaswani (Submitted on 12 Jun 2017, last revised 2 Aug 2023) <https://arxiv.org/abs/1706.03762>
* "Multivariate Probabilistic Time Series Forecasting with Informer" blog post by Eli Simhayev, Niels Rogge and Kashif Rasul (Published March 10, 2023) <https://huggingface.co/blog/informer>
