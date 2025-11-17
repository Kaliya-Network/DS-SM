# **Experiment 3: Perform Data Discovery, Profiling, and Create a Data Dictionary**

## **Aim**
To perform data discovery and profiling on a dataset, and to create a data dictionary that summarizes the structure, types, and properties of each attribute in the dataset.

---

## **Apparatus / Tools Required**
- System with Python (Jupyter Notebook / VS Code / Anaconda)
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
- Dataset: e.g., **iris.csv**, **sales_data_sample.csv**, or any structured dataset

---

## **Theory**

### **Data Discovery**
The initial exploration of a dataset to understand:
- Shape  
- Size  
- Basic structure  
- Data types  

### **Data Profiling**
Provides deeper insights such as:
- Summary statistics  
- Missing value counts  
- Number of unique values per column  

### **Data Dictionary**
A structured summary table containing:
- Column names  
- Data types  
- Non-null value counts  
- Number of unique entries  

These steps ensure better understanding of data quality, completeness, and relationships before performing analysis or modeling.

---

## **Steps / Procedure**

### **Step 1: Load Dataset**
- Load a CSV dataset using `pd.read_csv()`.
- Example: `iris.csv`

### **Step 2: Data Discovery**
- Use:
  - `.info()`
  - `.head()`
  - `.shape`
  - `.dtypes`
- Identify data structure and types.

### **Step 3: Data Profiling**
- Use `.describe()` for statistical summaries.
- Use `.isnull().sum()` for missing values.
- Use `.nunique()` to count unique values.

### **Step 4: Data Dictionary Creation**
Create a summary table containing:
- Column Name  
- Data Type  
- Non-null Count  
- Unique Values  
Then save it as **data_dictionary.xlsx**.

### **Step 5: Identify Distributions**
- Use `seaborn.histplot()` for each numerical column.
- Observe skewness, spread, and outliers.

### **Step 6: Identify Correlations**
- Use `.corr()` to calculate correlation.
- Use heatmap to visualize relationships.

### **Step 7: Save Documentation**
- Export data dictionary to **data_dictionary.xlsx**.

---

## **Program / Code**

```python
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
````

---

## **Result**

* The dataset was successfully explored and profiled.
* A data dictionary was created, listing all attributes with their types, non-null counts, and unique values.
* Distribution plots and a correlation heatmap were generated for numerical features.

---

## **Conclusion**

This experiment demonstrated comprehensive data discovery and profiling using Python.
The data dictionary provides a clear understanding of dataset structure and quality, which is essential before further analysis or model building.