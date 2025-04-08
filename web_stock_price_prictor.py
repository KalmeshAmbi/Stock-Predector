import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt

st.title("Stock Price Predictor App")

# Take stock ticker input from the user
stock = st.text_input("Enter the Stock ID", "GOOG")

# Set the date range for the stock data
end = datetime.now()
start = datetime(end.year-20, end.month, end.day)

# Download the stock data from Yahoo Finance
google_data = yf.download(stock, start=start, end=end)

# Handle empty data (invalid ticker or connection issues)
if google_data.empty:
    st.error(f"Could not retrieve data for ticker: {stock}. Please check the stock symbol.")
else:
    # Load the pre-trained LSTM Keras model
    model = load_model("Latest_stock_price_model.keras")

    # Display the stock data
    st.subheader("Stock Data")
    st.write(google_data)

    # Calculate and plot moving averages
    splitting_len = int(len(google_data) * 0.7)
    x_test = pd.DataFrame(google_data.Close[splitting_len:])

    def plot_graph(figsize, values, full_data, extra_data=0, extra_dataset=None):
        """ A function to plot stock price and moving averages """
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(values, 'orange', label='Moving Average')
        ax.plot(full_data.Close, 'b', label='Original Close Price')
        if extra_data:
            ax.plot(extra_dataset, label='Extra Dataset')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend(loc='upper left')
        ax.set_title("Stock Price and Moving Averages")
        plt.xticks(rotation=45)
        return fig

    # Plot different moving averages
    st.subheader('Original Close Price and MA for 250 days')
    google_data['MA_for_250_days'] = google_data.Close.rolling(250).mean()
    google_data.dropna(inplace=True)
    st.pyplot(plot_graph((15, 6), google_data['MA_for_250_days'], google_data))

    st.subheader('Original Close Price and MA for 200 days')
    google_data['MA_for_200_days'] = google_data.Close.rolling(200).mean()
    google_data.dropna(inplace=True)
    st.pyplot(plot_graph((15, 6), google_data['MA_for_200_days'], google_data))

    st.subheader('Original Close Price and MA for 100 days')
    google_data['MA_for_100_days'] = google_data.Close.rolling(100).mean()
    google_data.dropna(inplace=True)
    st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data))

    st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
    st.pyplot(plot_graph((15, 6), google_data['MA_for_100_days'], google_data, 1, google_data['MA_for_250_days']))

    # Scaling the data for prediction
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(google_data[['Close']])


    x_data = []
    y_data = []

    # Prepare data for prediction (100 days of data to predict the next day)
    for i in range(100, len(scaled_data)):
        x_data.append(scaled_data[i-100:i])
        y_data.append(scaled_data[i])

    x_data, y_data = np.array(x_data), np.array(y_data)

    # Make predictions using the model
    predictions = model.predict(x_data)

    # Inverse scaling for predictions and actual test data
    inv_pre = scaler.inverse_transform(predictions)
    inv_y_test = scaler.inverse_transform(y_data)

    # Calculate RMSE
    rmse = sqrt(mean_squared_error(inv_y_test, inv_pre))
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Prepare the data for display
    ploting_data = pd.DataFrame({
        'original_test_data': inv_y_test.reshape(-1),
        'predictions': inv_pre.reshape(-1)
    }, index=google_data.index[100:])
    # Display the original vs predicted values
    st.subheader("Original values vs Predicted values")
    st.write(ploting_data)

    # Plot the original and predicted closing prices
    st.subheader('Original Close Price vs Predicted Close Price')
    fig = plt.figure(figsize=(15, 6))
    plt.plot(pd.concat([google_data.Close[:splitting_len+100], ploting_data], axis=0), label='Original Data')
    plt.plot(ploting_data['predictions'], label='Predicted Data', linestyle='--', color='red')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title("Stock Price with Predictions")
    plt.grid(True)
    st.pyplot(fig)

    # Add a download button for predictions
    import io

    def convert_df_to_csv(df):
        """Convert the DataFrame to CSV format for downloading."""
        return df.to_csv().encode('utf-8')

    csv_data = convert_df_to_csv(ploting_data)
    st.download_button(
        label="Download Predictions as CSV",
        data=csv_data,
        file_name='stock_predictions.csv',
        mime='text/csv'
    )
