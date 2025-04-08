# Stock Price Prediction

A machine learning web application that predicts stock prices using historical data. The project incorporates a trained Keras model for forecasting future stock prices and is built with a user-friendly Streamlit interface.

## Overview

This project uses:
- **Streamlit** to create an interactive web application.
- **Keras** to load a pre-trained model (`Latest_stock_price_model.keras`) for making predictions.
- **yfinance** to download historical stock data.
- **Pandas, NumPy, and Matplotlib** for data manipulation and visualization.
- **Scikit-Learn** for data preprocessing and performance metrics.

The application allows users to input a stock ticker symbol (default is "GOOG"), fetches the corresponding historical data, and displays both the raw stock data and predictions from the model.


