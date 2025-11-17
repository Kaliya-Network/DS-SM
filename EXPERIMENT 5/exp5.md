# **Experiment 5: Transform Raw Data (Scaling, Encoding, Normalization, and Standardization)**

## **Aim**
To transform raw data for analysis by applying scaling, encoding, normalization, and standardization using Python and Scikit-learn.

---

## **Apparatus / Tools Required**
- Python (Jupyter Notebook / VS Code / Anaconda)
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`

---

## **Theory**

Data transformation is an essential preprocessing step to prepare data for analysis or machine learning.

### **1. Standardization**
Rescales features so they have:  
- Mean = 0  
- Standard deviation = 1  

### **2. Min-Max Scaling**
Transforms values to a specific range (commonly 0 to 1).

### **3. Encoding**
Converts categorical data into numerical format.  
Example: One-Hot Encoding.

### **4. Normalization**
Scales samples to have unit length (L2 norm = 1).

These transformations improve model accuracy, performance, and consistency.

---

## **Steps / Procedure**

### **1. Create a sample dataset**
Include numerical and categorical columns.

### **2. Apply transformations using Scikit-learn**
- `StandardScaler` → Standardization of **Age**  
- `MinMaxScaler` → Scaling of **Income**  
- `OneHotEncoder` → Encoding **City**  

### **3. Use ColumnTransformer**
Apply multiple transformations at the same time.

### **4. Apply L2 Normalization**
Normalize selected vectors using `Normalizer`.

### **5. Convert final outputs into a DataFrame**
For easier viewing and analysis.

---

## **Program / Code**

```python
# EXPERIMENT 5: DATA TRANSFORMATION USING SCIKIT-LEARN

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, Normalizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# 1. Create a sample dataset
data = {
    'Age': [25, 30, 60, 45, 22, 55],
    'Income': [20000, 150000, 35000, 80000, 25000, 95000],
    'City': ['New York', 'London', 'Paris', 'New York', 'London', 'Paris'],
    'Experience_Years': [1, 10, 30, 15, 1, 25]
}

df = pd.DataFrame(data)
print("--- Original Dataset ---")
print(df)
print("-" * 50)

# 2. Define transformations
numerical_features = ['Age', 'Income']
categorical_features = ['City']

age_transformer = StandardScaler()
income_transformer = MinMaxScaler()
city_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# 3. ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('standard_scaling', age_transformer, ['Age']),
        ('minmax_scaling', income_transformer, ['Income']),
        ('onehot_encoding', city_transformer, categorical_features)
    ],
    remainder='passthrough'
)

# 4. Apply transformations
transformed_data = preprocessor.fit_transform(df)
feature_names = (
    ['Age_Standardized', 'Income_Scaled'] +
    list(preprocessor.named_transformers_['onehot_encoding'].get_feature_names_out(categorical_features)) +
    ['Experience_Years']
)

df_transformed = pd.DataFrame(transformed_data, columns=feature_names)
print("--- Transformed Data ---")
print(df_transformed)
print("-" * 50)

# 5. Normalization (L2 norm)
print("--- Normalization Example ---")
sample_vector = df_transformed.iloc[0, :2].values.reshape(1, -1)
normalizer = Normalizer(norm='l2')
normalized_vector = normalizer.fit_transform(sample_vector)
print(f"Normalized vector: {normalized_vector}")
print(f"L2 Norm: {np.linalg.norm(normalized_vector[0]):.4f}")
````

---

## **Result**

The dataset was successfully transformed using multiple preprocessing techniques — standardization, scaling, encoding, and normalization — making it ready for further analysis or machine learning applications.

---

## **Output**

* **Age** → standardized (mean ≈ 0, std ≈ 1)
* **Income** → scaled between 0 and 1
* **City** → one-hot encoded into three binary columns
* **Normalization** applied on a sample vector → produced a unit L2 norm

---

## **Conclusion**

This experiment demonstrates data transformation techniques using Scikit-learn.
Scaling, encoding, standardization, and normalization help produce accurate and consistent data for analysis and machine learning workflows.
