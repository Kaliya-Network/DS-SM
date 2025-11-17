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