# EXPERIMENT 4: DATA CLEANING USING PANDAS AND NUMPY

# Import required libraries
import pandas as pd
import numpy as np

# STEP 1: LOAD THE DATASET
df = pd.read_csv('sales_data_sample_copy.csv', encoding='latin1')
print("Original Dataset (First 5 Rows):")
print(df.head())

# STEP 2: BASIC DATA INFORMATION
print("\n--- Dataset Information ---")
print(df.info())

print("\n--- Summary Statistics ---")
print(df.describe(include='all'))

# STEP 3: DETECT AND HANDLE MISSING VALUES
print("\n--- Checking for Missing Values ---")
print(df.isnull().sum())

# Fill missing numeric columns with mean
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Fill missing categorical columns with mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("\n--- Missing Values After Handling ---")
print(df.isnull().sum())

# STEP 4: DETECT AND REMOVE DUPLICATE RECORDS
print("\n--- Checking for Duplicate Records ---")
duplicate_count = df.duplicated().sum()
print(f"Number of duplicate records: {duplicate_count}")
df.drop_duplicates(inplace=True)
print("Duplicate records removed successfully.")

# STEP 5: HANDLE INCONSISTENT FORMATS
if 'CUSTOMERNAME' in df.columns:
    df['CUSTOMERNAME'] = df['CUSTOMERNAME'].str.title()

if 'SALES' in df.columns:
    df['SALES'] = df['SALES'].replace('[\$,]', '', regex=True).astype(float)

if 'ORDERDATE' in df.columns:
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')

# STEP 6: DISPLAY CLEANED DATA
print("\nCleaned Dataset (First 5 Rows):")
print(df.head())

print("\n--- Data Types After Cleaning ---")
print(df.dtypes)

# STEP 7: EXPORT CLEANED DATASET
df.to_csv("cleaned_sales_data.csv", index=False)
print("\nCleaned dataset has been exported as 'cleaned_sales_data.csv'.")
