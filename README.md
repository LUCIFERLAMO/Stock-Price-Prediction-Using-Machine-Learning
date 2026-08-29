# Stock-Price-Prediction-Using-Machine-Learning
A machine learning project that predicts the next trading day's High and Low stock prices using historical market data.

## Overview

This project scrapes historical stock data for **GMR Airports (NSE)** from [stockanalysis.com](https://stockanalysis.com), engineers time-series features from it, and trains a **Linear Regression** model to predict the next trading day's **High** and **Low** prices. Predictions are displayed through a simple **Tkinter GUI**.

## Features

- Web scraping with `requests` + `BeautifulSoup` to pull historical OHLC (Open, High, Low, Close) data
- Data cleaning and preprocessing with `pandas` (type conversion, sorting, handling missing values)
- Feature engineering using previous-day and next-day High, Low, and Close values
- Linear Regression model (via `scikit-learn`) trained separately for High and Low prediction
- Model evaluation using Mean Absolute Error (MAE)
- Simple desktop GUI (Tkinter) to display the next day's predicted High and Low

## Tech Stack

- Python
- pandas
- BeautifulSoup (bs4)
- scikit-learn
- Tkinter

## How It Works

1. Scrapes the last 50 trading days of GMR Airports stock data from the website
2. Parses the HTML table and extracts Date, Open, High, Low, Close, Adj Close, Change, and Volume
3. Cleans and converts the data into numeric/datetime formats
4. Sorts the data chronologically
5. Creates lag features (previous day's Close/High/Low) and target features (next day's High/Low)
6. Splits the data 80/20 into training and testing sets
7. Trains separate Linear Regression models to predict next-day High and next-day Low
8. Evaluates model accuracy using MAE
9. Displays the latest trading day and predicted next High/Low through a GUI

## Getting Started

### Prerequisites

pip install requests beautifulsoup4 pandas scikit-learn


### Run

python stock_predictor.py


The GUI will open, showing the latest trading day. Click **PREDICT** to see the predicted High and Low for the next trading day.

## Disclaimer

This project was built for **educational purposes only**, to practice web scraping, data preprocessing, and machine learning. It should **not** be used for real financial or trading decisions.
