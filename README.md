# Stock Price Prediction & Forecasting using LSTM Neural Networks

## Project Overview

This project demonstrates how to build and train a Long Short-Term Memory (LSTM) neural network to forecast stock prices. LSTMs are a type of Recurrent Neural Network (RNN) that excel in modeling sequential and time-series data. By leveraging 5 years of historical stock prices, this project aims to predict future stock movements and provide insights into potential trends.

## Objective

-Develop a deep learning model to forecast stock prices using 5 years of historical data.

-Utilize LSTM neural networks to capture temporal dependencies and trends in sequential stock data.

## Methodology
1. Data Collection

-Acquired 5 years of historical daily stock prices (e.g., Microsoft MSFT) using the yfinance Python library.

-Focused on daily closing prices for modeling and prediction.

2. Data Preprocessing

-Scaling: Applied MinMaxScaler to normalize data between 0 and 1 to improve training stability.

-Windowing: Created sequences of 60 previous days’ prices to predict the next day’s price.

-Train-Test Split: Divided dataset into 80% training and 20% testing sets to evaluate the model’s predictive performance on unseen data.

3. LSTM Model Architecture

-Input Layer: LSTM layer accepting 60-day price windows.

-Hidden Layers: Two LSTM layers to capture complex temporal patterns, followed by dense layers for feature extraction.

-Output Layer: Single-neuron dense layer for predicting the next day’s stock price.

4. Training

-Model trained on the training set using Mean Squared Error (MSE) loss and an appropriate optimizer.

-Early stopping and checkpoints implemented to prevent overfitting and save the best-performing model.

## Results

-Predictive Accuracy: Evaluated using Root Mean Squared Error (RMSE), demonstrating that predictions closely align with actual stock prices.

-Visualization: Generated plots comparing predicted vs. actual prices on the test set, showing the model’s ability to follow general trends.

-Successfully captured trends and short-term movements over a 5-year period, making it suitable for practical forecasting.

## Tools & Technologies

-Python
-yfinance for data acquisition

-pandas, numpy for data manipulation

-scikit-learn for scaling and preprocessing

-TensorFlow / Keras for building and training the LSTM model

-matplotlib / seaborn for visualization

## Key Takeaways

-LSTMs are highly effective for time-series forecasting due to their memory of previous states.

-Proper preprocessing, including scaling and windowing, is critical for model performance.

-Using 5 years of historical data improves the model’s ability to capture trends and seasonal patterns.

### Colab file link: https://colab.research.google.com/drive/186-T0_itLJ-I94i7IsA7yXTT2ldnxuGa?usp=sharing

<img width="1325" height="711" alt="res" src="https://github.com/user-attachments/assets/0336f1d0-39cc-4ca5-a6a0-dbd1f330c3bf" />

## Model Predictions Visualization(Insights)

-The plot shows the train dataset (blue), the actual stock prices from the test set (orange), and the model predictions (green).

-The model closely follows the trend of the actual prices, demonstrating its ability to forecast stock movements based on historical data.

-Using 5 years of historical stock prices, the LSTM successfully captures short-term trends and general patterns in stock price fluctuations.
