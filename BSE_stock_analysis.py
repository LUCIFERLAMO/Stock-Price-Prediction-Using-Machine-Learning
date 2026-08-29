import requests

url = "https://stockanalysis.com/quote/nse/GMRAIRPORT/history/"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)

if response.status_code == 200:
    print("Connected Successfully ")
else:
    print(f"Error cox of {response.status_code}.")
    exit(0)

# --- giving the response to beautiful soup

from bs4 import BeautifulSoup

soup = BeautifulSoup(response.text,"html.parser")

# ---- Finding all the tables 

tables = soup.find_all("table")
table = tables[0]

#print(table)


# ------- finding the table from the website and storing it in a vairable called rows ----


rows = table.find_all("tr")
print(f"Number of rows retrived: {len(rows)}") # 51 rows in total

# -------------- test ------ print the first 3 rows -----------

#for row in rows[1:4]:
#    cells = row.find_all("td")
#    print([cell.get_text(" ",strip=True) for cell in cells])

 # works ;)


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

print("Records Recived:", len(stock_data))

# --- works :)

# ---- Feading the recirds recived to pandas 

import pandas as pd

df = pd.DataFrame(stock_data, columns=["Date","Open","High","Low","Close","Adj_Close","Change","Volume"])



# --- converting the ojbject types into numeric data 

df["Date"] = pd.to_datetime(df["Date"])
df["Change"] = pd.to_numeric(df["Change"].str.replace("%",""))
df["Volume"] = pd.to_numeric(df["Volume"].str.replace(",",""))

numeric_column = ["Open","High","Low","Close","Adj_Close"]

for col in numeric_column:
    df[col] = pd.to_numeric(df[col])

# print(df.info())

# ---- sorting the data to assecnding order of date

df = df.sort_values("Date").reset_index(drop=True) # When resetting the index, throw away the old index instead of keeping it as a new column
#print(df)


# --- creating tmr high and tmr low

df["Next_high"] = df["High"].shift(-1)
df["Next_low"] = df["Low"].shift(-1)


df["Previous_Close"] = df["Close"].shift(1)


df["Previous_High"] = df["High"].shift(1)
df["Previous_Low"] = df["Low"].shift(1)

#print(df.head(3))
#print(df.tail(3))


# --- checking the data if its clean 


#print("Total rows:", len(df))
#print("\nDate range:")
#print("First date:", df["Date"].iloc[0])
#print("Last date:", df["Date"].iloc[-1])

#print("\nMissing values:")
#print(df.isnull().sum()) # Aug 28 {present day} doesn't have a future day's value yet. which is what we r predicting

#print("\nLast 5 rows:")
#print(df.tail())

# -- works 


# --- Making a sperate copy to train the model

model_data = df.dropna(subset=["Previous_Close","Previous_High","Previous_Low","Next_high","Next_low"]).copy() 

# y copy ? as we r making changes to our individual copy cox pandas might sometime give some error for this
# subset tells we want to drop all the NA values but only chekc for the 2 columns mentions 

latest_day = df.iloc[-1]

print("Training/testing records:", len(model_data))

#print("\nLatest trading day:")
#print(latest_day)

print("\nLatest date:", latest_day["Date"])
print("Latest High:", latest_day["High"])
print("Latest Low:", latest_day["Low"])


#  ------ we will now train the model using only some features 

# ---- Features 

x = model_data[["Open", "High", "Low", "Close","Previous_Close","Previous_High","Previous_Low"]]

# --- target (what we want the model to learn)

y_high = model_data["Next_high"]
y_low = model_data["Next_low"]



# ------ check ------

#print("Features given to the model:")
#print(x.columns.tolist())

#print("\nNumber of features:", x.shape[1])

#print("\nX shape:", x.shape)
#print("Y_high shape:", y_high.shape)
#print("Y_low shape:", y_low.shape)

#print("\nFirst 3 training examples:")
#print(x.head(3))

#print("\nCorresponding answers (Next High):")
#print(y_high.head(3))

#print("\nCorresponding answers (Next Low):")
#print(y_low.head(3))


# ---- train the model ----

# -- split the data 

split_data = int(len(x) * 0.8)

# ----features

# -- dividing the table 
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

# -- check 

#print("Total known records:", len(x))
#print("Training records:", len(x_train))
#print("Testing records:", len(x_test))

#print("\nTraining period:")
#print(train_dates.iloc[0], "→", train_dates.iloc[-1])

#print("\nTesting period:")
#print(test_dates.iloc[0], "→", test_dates.iloc[-1])

#print("\nTraining shape:", x_train.shape)
#print("Testing shape:", x_test.shape)


# ----- training the model

from sklearn.linear_model import LinearRegression

# --- high model

high_model_lr = LinearRegression()

high_model_lr.fit(x_train,y_high_train)

# --- low model

low_model_lr = LinearRegression()

low_model_lr.fit(x_train,y_low_train)

# ---- make the predictions

lr_high_prediction = high_model_lr.predict(x_test)

lr_low_prediction = low_model_lr.predict(x_test)

# --- check the MAE

from sklearn.metrics  import mean_absolute_error

high_mae_lr = mean_absolute_error(y_high_test,lr_high_prediction)

low_mae_lr = mean_absolute_error(y_low_test,lr_low_prediction)

print("=" *50)
print(f"High MAE: {high_mae_lr}")
print(f"low MAE: {low_mae_lr}")
print("=" *50)


# ---- the model gave us unhappy results 

# --- Adding previous close to the model 



#print(x)

# --- checking y we r facing very high MAE
#print("===== TRAINING PERIOD =====")

#print("High range:",
#      x_train["High"].min(),
#      "to",
#      x_train["High"].max())

#print("Low range:",
#      x_train["Low"].min(),
#      "to",
#      x_train["Low"].max())


#print("\n===== TESTING PERIOD =====")

#print("High range:",
#      x_test["High"].min(),
#      "to",
#      x_test["High"].max())

#print("Low range:",
#      x_test["Low"].min(),
#      "to",
#      x_test["Low"].max())


# we r facing an distribution shift = The data used to teach the model looks very different from the data we're asking it to predict.

# so we r trying to slove it by adding yesterdays high and low in the feature


# so now we r tryning to find what did the linear regression model learn from each model
print("===== HIGH MODEL =====")

for feature, coefficient in zip(x_train.columns, high_model_lr.coef_): 
# when u train the model (fit) the linear regression makes a co-effiecnt of every feature it has been trained on.
# So when u see smt like Previous_Close: 2.6083920393298408e-15 it says how much this feature affects the final output
    print(f"{feature}: {coefficient}")


print("\nIntercept:", high_model_lr.intercept_)
# intercept means - before we train the model we will have a starting value of the data we have and then from that data we reduce or add values based on our co efficents 


print("\n===== LOW MODEL =====")

for feature, coefficient in zip(x_train.columns, low_model_lr.coef_):
    print(f"{feature}: {coefficient}")


print("\nIntercept:", low_model_lr.intercept_)

print("=" * 70)

# things learnt

# having a negative coefficent, does not mean u have to remove it from the model training feature 
# having a very high coeffient and the rest of the co effiecnt r low means that the particular feature have big values compared to others
# so in order to improve the MAE u have to experiement by removing some features


# now by removing volumes which imporved significantly and removing change which also improved a bit we got the overall best MAE of High MAE: 0.6455492753423385    low MAE: 0.7539124433680044 best from the past linear model and random forest model

# -- seeing the predictions it makes 

print("===== HIGH PREDICTIONS =====")

for date, actual, predicted in zip(
    test_dates,
    y_high_test,
    lr_high_prediction
):
    print(
        f"{date.date()} | "
        f"Actual: {actual:.2f} | "
        f"Predicted: {predicted:.2f} | "
        f"Error: {abs(actual - predicted):.2f}"
    )


print("\n===== LOW PREDICTIONS =====")

for date, actual, predicted in zip(
    test_dates,
    y_low_test,
    lr_low_prediction
):
    print(
        f"{date.date()} | "
        f"Actual: {actual:.2f} | "
        f"Predicted: {predicted:.2f} | "
        f"Error: {abs(actual - predicted):.2f}"
    )