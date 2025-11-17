# **Experiment 1: Data Collection and Creation of DataFrame using Pandas**

## **Aim**
To collect data from different sources and create a DataFrame using the Pandas library in Python.

---

## **Apparatus / Tools Required**
- System with Python (Anaconda / Jupyter Notebook / VS Code)
- Required Libraries:
  - `pandas`
  - `requests`
  - `beautifulsoup4`

---

## **Theory**

Data collection is a crucial step in Data Science. Data can originate from various sources, such as:

### **1. Manual Data Entry**
Data created directly within Python using dictionaries or lists.

### **2. CSV Import**
Structured data stored in CSV files can be imported using `pd.read_csv()`.

### **3. Web API Access**
APIs provide data in formats like JSON. The `requests` library helps fetch online data.

### **4. Web Scraping**
Involves extracting information from webpages using tools like BeautifulSoup.

The **Pandas DataFrame** structure allows efficient representation, handling, and analysis of such tabular data.

---

## **Program / Code**

### **1. Manual Data Entry**
```python
import pandas as pd

manual_data = {
    'Name': ['Alice', 'Bob', 'Cathy'],
    'Age': [25, 30, 22],
    'City': ['New York', 'Los Angeles', 'Chicago']
}
df_manual = pd.DataFrame(manual_data)
print(df_manual)
````

---

### **2. CSV Import**

```python
import pandas as pd

df_csv = pd.read_csv('sample_data.csv')
print(df_csv)
```

---

### **3. Web API Access**

```python
import requests
import pandas as pd

api_url = 'https://jsonplaceholder.typicode.com/posts'
response = requests.get(api_url)

if response.status_code == 200:
    data_api = response.json()
    df_api = pd.DataFrame(data_api)
    print(df_api.head())
else:
    print("API request failed.")
```

---

### **4. Web Scraping**

```python
from bs4 import BeautifulSoup
import requests
import pandas as pd

url = 'https://quotes.toscrape.com/'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')

quotes = [quote.get_text() for quote in soup.find_all('span', class_='text')]
df_scrape = pd.DataFrame(quotes, columns=['Quote'])

print(df_scrape.head())
```

---

## **Result**

Data was successfully collected and converted into Pandas DataFrames using:

* Manual data entry
* CSV file import
* API data retrieval
* Web scraping

---

## **Conclusion**

Different sources provide different forms of data, and Pandas offers a powerful and flexible way to convert all of them into DataFrames. Whether the data comes from manual input, CSV files, APIs, or web scraping, Pandas makes it efficient and convenient to handle and analyze the data as required.

