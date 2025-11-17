import pandas as pd

manual_data = {
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['New York', 'Los Angeles', 'Chicago']
}

df_manual = pd.DataFrame(manual_data)
print(df_manual)