# Stock Price Predictor (GMR Airport)

A small end-to-end project that scrapes historical stock data for **GMR Airport (NSE)**, trains a machine learning model to predict the next trading day's **High** and **Low** prices, and displays the results in a desktop GUI with an embedded chart comparing actual vs predicted values.

## Features

- Scrapes live historical OHLC (Open, High, Low, Close) data from [stockanalysis.com](https://stockanalysis.com)
- Cleans and preprocesses the data using `pandas`
- Trains two separate `LinearRegression` models — one to predict next-day High, one for next-day Low
- Evaluates model accuracy using Mean Absolute Error (MAE)
- Predicts the next trading day's High/Low based on the latest available data
- Desktop GUI built with `Tkinter`
- Embedded `matplotlib` chart comparing actual vs predicted values on the test set

## Tech Stack

- Python 3
- `requests` + `BeautifulSoup4` — web scraping
- `pandas` — data cleaning and feature engineering
- `scikit-learn` — Linear Regression model + evaluation
- `matplotlib` — data visualization
- `tkinter` — GUI

## Installation

```bash
git clone https://github.com/LUCIFERLAMO/<your-repo-name>.git
cd <your-repo-name>
pip install requests beautifulsoup4 pandas scikit-learn matplotlib
```

> `tkinter` ships with most standard Python installations. On some Linux distros you may need to install it separately: `sudo apt install python3-tk`

## Usage

```bash
python stock_price_predictor.py
```

This will:
1. Scrape the latest historical price data
2. Train the High/Low prediction models
3. Open a GUI window showing:
   - The latest trading day
   - A **Predict** button showing the next day's predicted High/Low
   - Model accuracy (MAE)
   - A chart comparing actual vs predicted prices on the test set

## How It Works

1. **Scraping**: Historical price data is pulled from the stock's history page and parsed out of the HTML table.
2. **Feature Engineering**: For each day, the model uses that day's Open/High/Low/Close plus the *previous* day's Close/High/Low as features to predict the *next* day's High and Low.
3. **Train/Test Split**: Data is split chronologically (80% train / 20% test) — not shuffled, since this is time-series data.
4. **Model**: Two independent `LinearRegression` models are trained — one for High, one for Low.
5. **Visualization**: The test set's actual vs predicted High/Low values are plotted as two stacked line charts for easy visual comparison.

## Known Limitations

- Linear Regression assumes a simple linear relationship, which may not capture more complex market patterns
- Model accuracy depends heavily on the amount and quality of historical data scraped
- The scraper is tied to `stockanalysis.com`'s current HTML structure and may break if the site layout changes
- Currently hardcoded to a single stock (GMR Airport / NSE)

## Disclaimer

This project is for educational purposes only. Predictions are based on a simple linear model trained on limited historical data and should **not** be used as financial advice or for actual trading decisions.

## Author

**LUCIFERLAMO**
