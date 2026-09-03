import requests

url = "https://stockanalysis.com/quote/nse/GMRAIRPORT/history/"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    print("Connected Successfully ")
else:
    print(f"Error because of {response.status_code}.")
    exit(0)

# --- giving the response to beautiful soup

from bs4 import BeautifulSoup

soup = BeautifulSoup(response.text, "html.parser")

# ---- Finding all the tables

tables = soup.find_all("table")
table = tables[0]


# ------- finding the table from the website and storing it in a variable called rows ----

rows = table.find_all("tr")


# ---- Extracting all the 50 rows

stock_data = []

for row in rows[1:]:
    cells = row.find_all("td")

    date = cells[0].get_text(strip=True)
    open_price = cells[1].get_text(strip=True)
    high = cells[2].get_text(strip=True)
    low = cells[3].get_text(strip=True)
    close = cells[4].get_text(strip=True)
    adj_close = cells[5].get_text(strip=True)
    change = cells[6].get_text(strip=True)
    volume = cells[7].get_text(strip=True)

    stock_data.append([
        date,
        open_price,
        high,
        low,
        close,
        adj_close,
        change,
        volume
    ])


# ---- Feeding the records received to pandas

import pandas as pd

df = pd.DataFrame(
    stock_data,
    columns=[
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Adj_Close",
        "Change",
        "Volume"
    ]
)


# --- converting the object types into numeric data

df["Date"] = pd.to_datetime(df["Date"])
df["Change"] = pd.to_numeric(df["Change"].str.replace("%", ""), errors="coerce")
df["Volume"] = pd.to_numeric(df["Volume"].str.replace(",", ""), errors="coerce") # converts it into nan when an error appears

numeric_column = ["Open", "High", "Low", "Close", "Adj_Close"]

for col in numeric_column:
    df[col] = pd.to_numeric(df[col], errors="coerce")


# ---- sorting the data to ascending order of date

df = df.sort_values("Date").reset_index(drop=True)


# --- creating tomorrow's high and tomorrow's low

df["Next_high"] = df["High"].shift(-1)
df["Next_low"] = df["Low"].shift(-1)

df["Previous_Close"] = df["Close"].shift(1)
df["Previous_High"] = df["High"].shift(1)
df["Previous_Low"] = df["Low"].shift(1)


# --- Making a separate copy to train the model

model_data = df.dropna(
    subset=[
        "Open",
        "High",
        "Low",
        "Close",
        "Previous_Close",
        "Previous_High",
        "Previous_Low",
        "Next_high",
        "Next_low"
    ]
).copy()


latest_day = df.iloc[-1]

latest_date = latest_day["Date"]


#  ------ we will now train the model using only some features

# ---- Features

x = model_data[
    [
        "Open",
        "High",
        "Low",
        "Close",
        "Previous_Close",
        "Previous_High",
        "Previous_Low"
    ]
]


# --- target (what we want the model to learn)

y_high = model_data["Next_high"]
y_low = model_data["Next_low"]


# ---- train the model ----

# -- split the data

split_data = int(len(x) * 0.8)


# ----features

x_train = x.iloc[:split_data]
x_test = x.iloc[split_data:]


# -- dividing the next high

y_high_train = y_high.iloc[:split_data]
y_high_test = y_high.iloc[split_data:]


# -- dividing the next low

y_low_train = y_low.iloc[:split_data]
y_low_test = y_low.iloc[split_data:]


# -- dividing the dates

train_dates = model_data["Date"].iloc[:split_data]
test_dates = model_data["Date"].iloc[split_data:]


# ----- training the model

from sklearn.linear_model import LinearRegression

# --- high model

high_model_lr = LinearRegression()

high_model_lr.fit(x_train, y_high_train)


# --- low model

low_model_lr = LinearRegression()

low_model_lr.fit(x_train, y_low_train)


# ---- make the predictions

lr_high_prediction = high_model_lr.predict(x_test)

lr_low_prediction = low_model_lr.predict(x_test)


# --- check the MAE

from sklearn.metrics import mean_absolute_error

high_mae_lr = mean_absolute_error(
    y_high_test,
    lr_high_prediction
)

low_mae_lr = mean_absolute_error(
    y_low_test,
    lr_low_prediction
)


# -------------- giving the latest data to the model

latest_data = pd.DataFrame([{
    "Open": latest_day["Open"],
    "High": latest_day["High"],
    "Low": latest_day["Low"],
    "Close": latest_day["Close"],
    "Previous_Close": latest_day["Previous_Close"],
    "Previous_High": latest_day["Previous_High"],
    "Previous_Low": latest_day["Previous_Low"]
}])


# --- predict the next day's High

predicted_high = high_model_lr.predict(latest_data)[0]


# --- predict the next day's Low

predicted_low = low_model_lr.predict(latest_data)[0]


# ---------------- TKINTER GUI ----------------

import tkinter as tk

root = tk.Tk()

root.title("Stock Price Predictor")
root.geometry("800x1000")


# ---------------- TITLE ----------------

title_label = tk.Label(
    root,
    text="STOCK PRICE PREDICTOR",
    font=("Arial", 22, "bold")
)

title_label.pack(pady=20)


# ---------------- STOCK NAME ----------------

stock_name = tk.Label(
    root,
    text="GMR Airport",
    font=("Arial", 16)
)

stock_name.pack(pady=5)


# ---------------- LATEST DATE ----------------

day = tk.Label(
    root,
    text=f"Latest Trading Day: {latest_date.strftime('%d-%b-%Y')}",
    font=("Arial", 12)
)

day.pack(pady=5)


# ---------------- BUTTON FUNCTION ----------------

def predict():

    result_label.config(
        text=f"Predicted High: ₹{predicted_high:.2f}\n"
             f"Predicted Low: ₹{predicted_low:.2f}"
    )


# ---------------- PREDICT BUTTON ----------------

predict_button = tk.Button(
    root,
    text="PREDICT",
    font=("Arial", 14, "bold"),
    command=predict
)

predict_button.pack(pady=15)


# ---------------- RESULT ----------------

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 14)
)

result_label.pack(pady=5)


# ---------------- MODEL ACCURACY ----------------

accuracy_label = tk.Label(
    root,
    text=f"High MAE: ₹{high_mae_lr:.2f}\n"
         f"Low MAE: ₹{low_mae_lr:.2f}",
    font=("Arial", 11)
)

accuracy_label.pack(pady=5)


# ---------------- CHART: ACTUAL vs PREDICTED ----------------

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

fig, (ax_high, ax_low) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)

# --- High subplot ---
ax_high.plot(test_dates, y_high_test.values, label="Actual High",
             color="green", marker="o", linewidth=2)
ax_high.plot(test_dates, lr_high_prediction, label="Predicted High",
             color="orange", marker="x", linewidth=2, linestyle="--")
ax_high.set_title("High: Actual vs Predicted")
ax_high.set_ylabel("Price (₹)")
ax_high.grid(True, alpha=0.3)
ax_high.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

# --- Low subplot ---
ax_low.plot(test_dates, y_low_test.values, label="Actual Low",
            color="red", marker="o", linewidth=2)
ax_low.plot(test_dates, lr_low_prediction, label="Predicted Low",
            color="blue", marker="x", linewidth=2, linestyle="--")
ax_low.set_title("Low: Actual vs Predicted")
ax_low.set_xlabel("Date")
ax_low.set_ylabel("Price (₹)")
ax_low.grid(True, alpha=0.3)
ax_low.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0)

ax_low.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b"))
fig.autofmt_xdate()
fig.tight_layout()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(pady=10, fill="both", expand=True)


# ---------------- RUN GUI ----------------

root.mainloop()