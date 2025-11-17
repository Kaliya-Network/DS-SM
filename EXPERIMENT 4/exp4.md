# **Experiment 4: Data Cleaning using Pandas and NumPy**

## **Aim**
To clean and preprocess a given dataset by handling missing values, duplicates, inconsistent formats, and exporting a cleaned version using Pandas and NumPy in Python.

---

## **Apparatus / Tools Required**
- System with Python installed (Jupyter Notebook / VS Code / Anaconda)
- Libraries:
  - `pandas`
  - `numpy`
- Dataset: **sales_data_sample_copy.csv**

---

## **Theory**

Data cleaning is a crucial preprocessing step where raw data is converted into a structured, consistent, and error-free format.  
It includes:

### **1. Handling Missing Values**
- Filling numeric values with the mean  
- Filling categorical values with the mode  

### **2. Removing Duplicates**
- Detect duplicate rows  
- Remove duplicates using `drop_duplicates()`  

### **3. Fixing Inconsistent Formats**
- Converting text to consistent case  
- Cleaning numeric fields (removing symbols like `$` and `,`)  
- Converting date fields to standard datetime format  

### **4. Exporting Cleaned Data**
- Saving the corrected dataset for further use  

Pandas and NumPy provide powerful tools for efficient data manipulation during this process.

---

## **Steps / Procedure**

### **Step 1: Load the Dataset**
- Load CSV using `pd.read_csv()`  
- Use encoding `"latin1"` for special characters  

### **Step 2: View Basic Information**
- Use `.info()` and `.describe()`  

### **Step 3: Detect and Handle Missing Values**
- Fill numeric missing values with mean  
- Fill categorical missing values with mode  

### **Step 4: Detect and Remove Duplicates**
- Identify duplicates using `.duplicated()`  
- Remove using `.drop_duplicates()`  

### **Step 5: Handle Inconsistent Formats**
- Proper-case customer names  
- Clean the sales column by removing `$` and `,`  
- Convert date to `datetime` format  

### **Step 6: Display Cleaned Data**
- Use `.head()`  
- Verify data types  

### **Step 7: Export Cleaned Dataset**
- Save cleaned file as **cleaned_sales_data.csv**

---

## **Program / Code**

```python
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
````

---

## **Result**

The dataset was successfully cleaned by filling missing values, removing duplicates, correcting inconsistent data formats, and exporting the final cleaned dataset for further analysis.

---

## **Conclusion**

This experiment demonstrates how to clean and preprocess raw data efficiently using Pandas and NumPy.
Data cleaning ensures quality, consistency, and accuracy, which are essential for data analysis and visualization in data science workflows.
