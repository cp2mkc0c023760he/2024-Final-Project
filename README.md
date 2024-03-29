# Multivariable Time Series Forecasting for Trading Abuse with Deep Learning

## UPC AIDL-2024-Final-Project

This repository is the final project for the [UPC School's Artificial Intelligence with Deep Learning postgraduate course (2023-2024)](https://www.talent.upc.edu/ing/estudis/formacio/curs/310400/postgraduate-course-artificial-intelligence-deep-learning/). The team members are composed of:

* Juanjo Nieto (Advisor)
* Fernando G.
* Inés Badia
* Oscar Santillana
* Toni Lopez

## Project Overview

In the world of financial markets, the ability to predict price movements accurately is crucial, especially in mitigating trading abuses. An 'abuser' in our context refers to someone who exploits information or technology to gain an unfair advantage, ultimately profiting at the expense of others. If someone as an abuser develops a deep learning model for price prediction, he gains an unfair advantage in the market, putting trading companies at a disadvantage.

Our goal with is to develop forecasting models capable of predicting market prices in order to create strategies and measures to mitigate this problem, thereby safeguarding trading companies from such abuses.

## Custom LSTM vs Transformer-Informer Comparision

**LSTM**, is a type of recurrent neural network (RNN), offers a powerful framework for modeling sequential data. With their ability to capture long-term dependencies and remember past information, LSTM architectures excel in analyzing time series data such as financial markets. In this project, we explore the efficacy of LSTM networks in forecasting forex prices, harnessing their capacity to learn intricate patterns and dynamics inherent in market behavior.

![LSTM - How it works](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/126424332/1a0a88d5-536d-4e8b-9a6b-972625f6ac92)
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

![Transformers - How it works](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/126424332/feeaed92-f661-4548-a846-ca5c381098e2)

Source: [“Attention is all you need” paper by Vaswani, et al., 2017](https://arxiv.org/abs/1706.03762)

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
* Accuracy: A metric used in classification problems in deep learning, which measures how accurately the model's predictions match the true responses. In our case, it reflects the accuracy of predicting market trends. We introduced Direction Accuracy to assess if the predictions align with the actual values' directions. Additionally, we utilized another accuracy measure to determine if the predictions are within 1% of the actual values.

## Computational resources

* Runpod Cloud - Custom LSTM
  * System equipped with GPU RTX 4090

* Colab Pro - Transformer-Informer
  * GPU NVIDIA T4 Tensor

## Milestones

### M0. Research

Preliminary research on LSTM and Transformer architectures for time series forecasting.

### M1. Custom dataset

Creating a custom dataset tailored to the particular requirements from our project, including collecting public data and preprocessing historical price data for EUR pairs in ticks each 10 mins.

### M2. Selection of symbols to predict (EUR pairs)

Choosing the specific currency pairs (EUR pairs) that will be the focus of the price prediction models since they operate in same market hours.

### M3. Initial versions of LSTM & Transformer

Adapting the base of our Transfomer arquitecture since [Informer](https://huggingface.co/blog/informer) and in the LSTM case from pytorch.org.

### M4. Hyper parameter tuning

Fine-tuning the hyperparameters of the LSTM and Transformer models to optimize their performance and accuracy.

### M5. Final version and backtesting of the models

Refinement both models and conducting backtesting to assess their performance and validate their effectiveness.

### M6. Results comparison between models

Comparing the results obtained of both models to determine which architecture performs better for the task of price prediction.

### M7. Evaluate different symbols (not achieved)

For assessing their correlation and determine which one is the most predictable.

## Run the models

### LSTM

#### Train a forex symbol

1. Create currencies csv file:
``cd Data/ && cat currencies.csv.gza* > currencies.csv.gz && gunzip currencies.csv.gz``
2. Execute ``python main.py --option 1 --ticker {ticker}``
3. Output a model weights file.

#### Inference

1. Execute ``python main.py --option 2 --ticker {ticker} --model_path path``
2. Output an image in Output/images and ticker metrics.

#### Cross Validation + Backtesting

1. Execute ``python main.py --option 3 --ticker {ticker}``
2. Output avg metrics from different folds.

### Transformer - Informer

The code performs training, inference, cross-validation, and generates all statistics simply by executing the Colab.
We recommend using a T4 system, and it typically completes in around 3 hours.

## LSTM Results

In the presented findings, we have established a threshold by designating currency pairs with a cross-validation accuracy metric surpassing 50% as the cutoff point.

![LSTM - Results](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/122474104/14118ba5-fd85-47dd-bd48-1a79bb7ac294)

![Direction Accuarcy VS 1% Accuarcy](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/126424332/7563e90e-ff6e-4e0a-9b72-9b64852f83bf)

![EURAUD and EURUSD Symbol](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/126424332/88acbfbc-4bf3-4247-a3b3-d651c2cd9f67)

## Transformer Informer Results

In these results, we have considered as a cutoff point the currency pairs with a value above 50% both in the cross-validation and in the test for the accuracy metric. The scenario where one of them, cross-validation or test, are below 50%, the result is considered invalid for our analysis.

![Transformer - Results](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/122474104/b821c692-1c78-4707-af81-d0074606cd5c)

![EURUSD Symbol](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/126424332/3e19ef02-8768-497b-b24a-c27a170e93a7)

## Results - Comparative

In the 18 EUR currency pairs analized, only the two premises mentioned previously (cross validation & test > 50%) have been met in one pair, EURUSD, in the Transformer-Informer case and in four pairs in LSTM case. In this summary table, we can confirm that it seems our LSTM model performs better than the Transformer.

![Comparative](https://github.com/cp2mkc0c023760he/2024-Final-Project/assets/122474104/a60e4d38-9cfa-4a4f-a675-5a8a0724ca2c)

## Conclusions

* Gained experience applying LSTM/Transformer models for Time Series Forecasting
* Backtesting, pipeline alignment and custom metrics help to compare model results
* For the symbols chosen our LSTM model performs better than the Transformer:
  * They could predict a few symbols pairs (5/18): EUR ( AUD|NOK|NZD|HKD|USD )
    * Test accuracy (direction): between 50 % and 71 %
    * Cross validation accuracy (direction): greater than 50 %
* Our models:
  * Seem to detect different patterns (different best predicted symbols)
  * Require more testing on other symbols to evaluate its performance

* Creating a dataset is a complex task
* Feature engineering, since financial data is complex, needs to be researched
* Classical ML algorithms (improved with XGBoost) might perform better
* Any operation become CPU/GPU intensive when dealing with huge datasets and must be optimized
* Working with git earlier to showcase progress and standardize steps

## References

* "Understanding LSTM Networks" blog post by Christopher Olah (Posted on August 27, 2015) <https://colah.github.io/posts/2015-08-Understanding-LSTMs/>
* "Attention is all you need" paper by Vaswani (Submitted on 12 Jun 2017, last revised 2 Aug 2023) <https://arxiv.org/abs/1706.03762>
* "Multivariate Probabilistic Time Series Forecasting with Informer" blog post by Eli Simhayev, Niels Rogge and Kashif Rasul (Published March 10, 2023) <https://huggingface.co/blog/informer>
