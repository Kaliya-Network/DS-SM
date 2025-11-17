import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load the dataset
df = pd.read_csv("iris.csv")

# Step 2: Data Discovery
print("===== BASIC INFORMATION ABOUT DATA =====")
print(df.info())
print("\n===== FIRST FIVE ROWS =====")
print(df.head())
print("\n===== SHAPE OF DATA (ROWS, COLUMNS) =====")
print(df.shape)
print("\n===== DATA TYPES =====")
print(df.dtypes)

# Step 3: Data Profiling
print("\n===== SUMMARY STATISTICS =====")
print(df.describe(include='all'))
print("\n===== MISSING VALUES =====")
print(df.isnull().sum())
print("\n===== UNIQUE VALUE COUNT PER COLUMN =====")
print(df.nunique())

# Step 4: Data Dictionary Creation
data_dict = pd.DataFrame({
    'Column Name': df.columns,
    'Data Type': df.dtypes.values,
    'Non-Null Count': len(df) - df.isnull().sum().values,
    'Unique Values': df.nunique().values
})
print("\n===== DATA DICTIONARY =====")
print(data_dict)

# Step 5: Identify Distributions
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.show()

# Step 6: Identify Correlations
correlation_matrix = df.corr(numeric_only=True)
print("\n===== CORRELATION MATRIX =====")
print(correlation_matrix)

plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Step 7: Save Data Dictionary
data_dict.to_excel("data_dictionary.xlsx", index=False)
print("\nData Dictionary saved as 'data_dictionary.xlsx'")

