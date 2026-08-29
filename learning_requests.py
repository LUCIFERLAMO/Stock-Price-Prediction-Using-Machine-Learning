import requests
from bs4 import BeautifulSoup


# -------------------- Requesting the website from the server and stroing the data in a variable named response ------------------------------------

url = "https://www.indiainfoline.com/company/gmr-airports-ltd-historical-data"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9"
}

response = requests.get(url,headers=headers)

print(response.status_code)
#print(response.text[:1000])


# ----------------------- passing the data to beautiful soup -----------------------------

soup = BeautifulSoup(response.text,"html.parser") #takes that messy HTML and organizes it so Python can easily understand and search it.

# print(soup.title) # Find the <title> of this webpage and print it

tables = soup.find_all("table")
#print(f"Number of tables {len(tables)}")


# choosing the first table

table = tables[0] 

# getting the contents of the table and removing unwanted tags

values = soup.get_text(" ",strip=True)[:2000]

# soup.get_text -> takes only the text from the content it is given 
# " " -> it adds space btw each word it finds
# strip=True -> it removes the unwanted newline space btw the word
# [:2000] -> we dont want the entire content just give the first 2000 words

#print(values)

# ------------ finding the number of rows using tr (number of rows) --------------------------

rows = soup.find_all("tr")
# print(rows[1]) # selecting and print the first row using 1 not 0 as 0 will show all the data 



# ---------- taking the contents of the first row -----------------

content_of_cell = soup.find_all("td")
#print(len(content_of_cell))

# printing the contents

#for content in content_of_cell:
    #print(content.get_text(" ",strip=True))


# taking ever cell which will be like <td> 100 </td>
# then only extrcting the content and leaving the word so <td> 100 </td> )



# -----------------------------------


# now we r taking onlt the td values so that we can iterate through each value seprately properly 
cells = rows[1].find_all("td")


#0 → Date
#1 → Open
#2 → High
#3 → Low
#4 → Close

#date = cells[0].get_text(strip=True) # getting the text in the first td value without tags
#high = cells[2].get_text(strip=True)
#low = cells[3].get_text(strip=True)

#print("Date:", date)
#print("High:", high)
#print("Low:", low)


# ------------- printing the date, high, lows of all the rows we found ------------------



dates_and_high_and_low = []
for row in rows[1:]: # we r satrting from 1 as 0 is the header

    cell = row.find_all("td") # now we can get the idividual values of the row as cell will not hold like (<td> 100 <td>, <td> 500 <td>, etc)

    date = cell[0].get_text(strip=True)
    high = cell[2].get_text(strip=True)
    low = cell[3].get_text(strip=True)

    # adding more features to imporve the prediction

    open_price = cell[1].get_text(strip=True)
    close = cell[4].get_text(strip=True)
    trades = cell[5].get_text(strip=True)

    #print(f"date: {date}, high: {high}, low: {low}")
    dates_and_high_and_low.append([date,open_price,high,low,close,trades])





# now passing the list and making it into a data frame

import pandas as pd

df = pd.DataFrame(dates_and_high_and_low,columns=["Date","open_price","high","low","close","trades"])
#print(df)


#print(df.info())


# ---------- converting the string high and low to numarice numbers -------------

df["high"] = pd.to_numeric(df["high"])
df["low"] = pd.to_numeric(df["low"])
df["Date"] = pd.to_datetime(df["Date"]) # converting it as an actual date object so that python can handle it as an date 

df["open_price"] = pd.to_numeric(df["open_price"])
df["close"] = pd.to_numeric(df["close"])
df["trades"] = pd.to_numeric(df["trades"].str.replace(",",""))

#print(df.info())


# checking if the data is clean 

#print(df.head())
#print("\nNumber of records:", len(df))
#print("\nMissing values:")
#print(df.isnull().sum()) # sums up if there r any missing values in each column and shows it here


# done for the day 



# sorting the values in ascending order 

df = df.sort_values(["Date"])
#print(df)

# now we r adding a new column called next_high

df["Next_high"] = df["high"].shift(-1) # it removes the top value and appends the value below the top to the top, and the last value below that column will be nil
df["Next_Low"] = df["low"].shift(-1)

# why r we doing this ?

# as we r doing supervised learning where we give the question and the label for it
# we r proving the question what is the stock price for monday and we r giving the answer for it so that it learns the pattern

#print(df)

# now we have a nan value in the botton of the 2 new columns we created so we will remove them

df = df.dropna()

#print(df.head())
#print(df.columns)

# ------------------- this is what we will give to the model ----------


# this is the high and low we will show to the model stored in a variable called x
x = df[["open_price", "high", "low", "close", "trades"]]


# the high and low values which the mmodel will use to learn the pattern from its prediction to the actual value
Y_high = df["Next_high"]
Y_low = df["Next_Low"]



# ------------------ now we will divide the data into 2 halfs (one for training and one for testing) ---------------'



# Training data → the model learns patterns from it.
# Testing data → we give the model data it hasn't seen before and check how accurate its predictions are

split_data = int(len(x) * 0.8) # taking 80 ercentage of the data as training and rest 20 as testing

# so if we have 100 rows then we will take 80 of them for training the model and rest 20 for testing 


# spliting the data 
x_train = x.iloc[:split_data] 
x_test = x.iloc[split_data:]

# spliting the high
y_high_train = Y_high.iloc[:split_data] 
y_high_test = Y_high.iloc[split_data:] # same as y_high[:split_data] we r using iloc jist to specifically tell we want to split suing the index

# spliting the low
y_low_train = Y_low.iloc[:split_data]
y_low_test = Y_low.iloc[split_data:]

#training and testing dates

Train_date = df["Date"].iloc[:split_data]
Test_date = df["Date"].iloc[split_data:]


#print("===== TRAINING DATA =====")
#print(x_train)

#print("\n===== TESTING DATA =====")
#print(x_test)

#print("\n===== TRAINING TARGET: NEXT HIGH =====")
#print(y_high_train)

#print("\n===== TESTING TARGET: NEXT HIGH =====")
#print(y_high_test)

#print("\n===== TRAINING TARGET: NEXT LOW =====")
#print(y_low_train)

#print("\n===== TESTING TARGET: NEXT LOW =====")
#print(y_low_test)

# --------------------------------------------------------
# we r using regression model
# A regression model is a machine-learning model used to predict a numerical value.

from sklearn.linear_model import LinearRegression

# Create a Linear Regression model for predicting tomorrow's High, think of it as a empty student   
high_model = LinearRegression()

#print(high_model)


# -------------------- Training to predict for the high value ----------------------

# so to train the model we use fit(data,label)

high_model.fit(x_train,y_high_train) # its like saying this is the data to the mode and this is answer now find the relation btw them
#print("High prediction model trained successfully.")


# now we r seeing how well the model predicts

model_predictions = high_model.predict(x_test)

#print("Actual values:")
#print(y_high_test)

#print("predicted values:")
#print(model_predictions)

# ----------------------------- training to predict for the lower value -------------------

lower_model = LinearRegression()

# teach the model

lower_model.fit(x_train,y_low_train)
#print("lower Trained")

# predicting the values and comparing it with the actual results

predict_lower_values = lower_model.predict(x_test)

#print("Actual Data:")
#print(y_low_test)


#print("predicted values:")
#print(predict_lower_values)

# ----------------------------------------------------------------------

# now that we have trained out model we need to see how much error did the model make compared to the actual value

# we will use Mean Absolute Error (MAE).
# MAE tells us, on average, how many rupees our prediction was away from the actual value.

from sklearn.metrics import mean_absolute_error

# syntax = mean_absolute_error(test_data,model_predictions)

 # calculate the average error for High predictions

high_mae = mean_absolute_error(y_high_test,model_predictions)


# calculate the average error for low predictions

low_mae = mean_absolute_error(y_low_test,predict_lower_values)



print("High prediction MAE:", high_mae)
print("Low prediction MAE:", low_mae)

# creating a dataframe to see the date the actual high and the prdicted high and low and predicted low for good refference 

result = pd.DataFrame({
    "Date":Test_date,
    "Actual high":y_high_test,
    "Predicted high":model_predictions,
    "Actual low":y_low_test,
    "Predicted low":predict_lower_values
})

#print(result)


