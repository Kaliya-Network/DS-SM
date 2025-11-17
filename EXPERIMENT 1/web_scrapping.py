import pandas as pd
from bs4 import BeautifulSoup
import requests

url = 'https://quotes.toscrape.com/'
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
quotes = [quote.get_text() for quote in soup.find_all('span', class_='text')]
df_scrape = pd.DataFrame(quotes, columns=['Quote'])
print(df_scrape.head())