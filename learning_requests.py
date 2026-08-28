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
print(f"Number of tables {len(tables)}")


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
print(len(content_of_cell))

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

    #print(f"date: {date}, high: {high}, low: {low}")
    dates_and_high_and_low.append([date,high,low])



# now passing the list and making it into a data frame

import pandas as pd

df = pd.DataFrame(dates_and_high_and_low,columns=["Date","high","low"])
print(df)


print(df.info())


# ---------- converting the string high and low to numarice numbers -------------

df["high"] = pd.to_numeric(df["high"])
df["low"] = pd.to_numeric(df["low"])
df["Date"] = pd.to_datetime(df["Date"]) # converting it as an actual date object so that python can handle it as an date 

print(df.info())


# checking if the data is clean 

print(df.head())
print("\nNumber of records:", len(df))
print("\nMissing values:")
print(df.isnull().sum()) # sums up if there r any missing values in each column and shows it here


# done for the day 



# sorting the values in ascending order 

df = df.sort_values(["Date"])
print(df)

# now we r adding a new column called next_high

df["Next_high"] = df["high"].shift(-1) # it removes the top value and appends the value below the top to the top, and the last value below that column will be nil
df["Next_Low"] = df["low"].shift(-1)

# why r we doing this ?

# as we r doing supervised learning where we give the question and the label for it
# we r proving the question what is the stock price for monday and we r giving the answer for it so that it learns the pattern
print(df)


