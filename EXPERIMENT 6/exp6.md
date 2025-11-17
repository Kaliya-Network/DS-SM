# **Experiment 6: Detect and Treat Outliers and Inconsistent Entries using IQR and Z-score Methods**

## **Aim**
To detect and treat outliers and inconsistent entries in a dataset using IQR and Z-score methods, visualize them, apply Winsorization, and document the changes.

---

## **Apparatus / Tools Required**
- Python (Jupyter Notebook / VS Code)
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scipy.stats`

---

## **Theory**

Outliers are data points that deviate significantly from the rest of the dataset.  
They can negatively affect statistical analysis and machine learning models.

### **Outlier Detection Methods**
#### **1. IQR Method**
- IQR = Q3 − Q1  
- Outliers are values outside:  
  **[Q1 − 1.5 × IQR, Q3 + 1.5 × IQR]**

#### **2. Z-score Method**
- Z = (x − mean) / std  
- Outliers if **|Z| > 3**

### **Outlier Treatment Methods**
- **Winsorization:** Replace extreme values with lower/upper percentile limits  
- **Transformation:** Apply log or scaling to reduce outlier impact  

### **Visualization**
Plots such as:
- Boxplots  
- Histograms  
help identify and validate outlier treatment.

---

## **Steps / Procedure**

### **1. Import Required Libraries**
Use pandas, numpy, matplotlib, scipy.stats.

### **2. Load Dataset**
Use a dataset containing intentional outliers.

### **3. Detect Outliers using IQR**
Calculate Q1, Q3, IQR, and detect out-of-range values.

### **4. Detect Outliers using Z-score**
Compute Z-scores and identify records with |Z| > 3.

### **5. Visualize Outliers**
Generate:
- Boxplots  
- Histograms  

### **6. Apply Winsorization**
Replace extreme values using the 5th and 95th percentile caps.

### **7. Re-visualize Data**
Confirm the effect of treatment.

### **8. Document Changes**
Add a change flag for modified rows.

---

## **Program / Code**

```python
# EXPERIMENT 6: Detect and Treat Outliers and Inconsistent Entries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Step 1: Load dataset
data = pd.DataFrame({
    'Student_ID': range(1, 11),
    'Marks': [45, 47, 49, 50, 52, 55, 90, 95, 100, 102]  # Intentional outliers
})
print("Original Data:\n", data)

# Step 2: Detect outliers using IQR method
Q1 = data['Marks'].quantile(0.25)
Q3 = data['Marks'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
outliers_iqr = data[(data['Marks'] < lower_limit) | (data['Marks'] > upper_limit)]
print("\nOutliers detected using IQR method:\n", outliers_iqr)

# Step 3: Detect outliers using Z-score method
z_scores = np.abs(stats.zscore(data['Marks']))
outliers_z = data[z_scores > 3]
print("\nOutliers detected using Z-score method:\n", outliers_z)

# Step 4: Visualization before treatment
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.boxplot(data['Marks'])
plt.title("Before Treatment (Boxplot)")

plt.subplot(1, 2, 2)
plt.hist(data['Marks'], bins=8, edgecolor='black')
plt.title("Before Treatment (Histogram)")
plt.tight_layout()
plt.show()

# Step 5: Winsorize outliers
lower_cap = data['Marks'].quantile(0.05)
upper_cap = data['Marks'].quantile(0.95)
data['Marks_Winsorized'] = np.where(data['Marks'] > upper_cap, upper_cap,
                                    np.where(data['Marks'] < lower_cap, lower_cap, data['Marks']))

# Step 6: Visualization after Winsorization
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.boxplot(data['Marks_Winsorized'])
plt.title("After Winsorization (Boxplot)")

plt.subplot(1, 2, 2)
plt.hist(data['Marks_Winsorized'], bins=8, edgecolor='black')
plt.title("After Winsorization (Histogram)")
plt.tight_layout()
plt.show()

# Step 7: Document changes
data['Change_Flag'] = np.where(data['Marks'] != data['Marks_Winsorized'], 'Modified', 'Unchanged')
print("\nDocumented Changes:\n", data)
````

---

## **Result**

* Outliers were detected using both IQR and Z-score methods.
* Winsorization was applied to treat extreme values.
* Visualizations confirmed improved distribution.
* Modified entries were flagged for documentation.

---

## **Output**

The program displayed:

* Outliers identified using IQR and Z-score
* Boxplots and histograms before and after treatment
* A cleaned dataset with adjusted values
* A new `Change_Flag` column indicating modified records

---

## **Conclusion**

The experiment successfully detected and treated outliers using statistical techniques.
Visualization validated the results, and Winsorization produced a more consistent and reliable dataset for further analysis.
