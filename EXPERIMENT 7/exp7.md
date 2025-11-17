# **Experiment 7: Perform Data Integration and Reduction (Merging and Dimensionality Reduction)**

## **Aim**
To perform data integration by merging multiple datasets and apply dimensionality reduction using Principal Component Analysis (PCA) for simplifying data while retaining important information.

---

## **Apparatus / Tools Required**
- Python (Jupyter Notebook / VS Code)
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `scikit-learn`

---

## **Theory**

### **Data Integration**
Combines data from multiple sources to create a unified dataset suitable for analysis.

### **Dimensionality Reduction**
Reduces the number of variables while preserving most of the useful information.

### **Standardization**
Ensures features have:
- Mean = 0  
- Variance = 1  

### **PCA (Principal Component Analysis)**
Transforms correlated features into a smaller set of uncorrelated variables called **Principal Components**.

PCA helps:
- Reduce dimensionality  
- Remove redundancy  
- Improve model performance  

---

## **Steps / Procedure**

### **1. Import Libraries**
Use pandas, numpy, matplotlib, and scikit-learn.

### **2. Create or Load Sample Datasets**
Example:  
- Student details  
- Marks dataset  

### **3. Perform Data Integration**
Use `pd.merge()` to merge datasets on `Student_ID`.

### **4. Select Numeric Columns**
Choose features for scaling and PCA.

### **5. Apply Standardization**
Use `StandardScaler()` to normalize numerical features.

### **6. Apply PCA**
Reduce dataset to **two principal components**.

### **7. Combine Reduced Data**
Attach PC1 and PC2 to original important columns.

### **8. Visualize Results**
Plot PC1 vs PC2 in a scatter plot with labels.

### **9. Display Explained Variance**
Show contribution of each principal component.

---

## **Program / Code**

```python
# EXPERIMENT 7: DATA INTEGRATION AND DIMENSIONALITY REDUCTION

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Step 1: Create sample datasets
student_details = pd.DataFrame({
    'Student_ID': [1, 2, 3, 4, 5],
    'Name': ['Amit', 'Bhavna', 'Chetan', 'Divya', 'Esha'],
    'Department': ['CSE', 'CSE', 'ECE', 'ECE', 'ME']
})

marks_details = pd.DataFrame({
    'Student_ID': [1, 2, 3, 4, 5],
    'Maths': [85, 78, 92, 88, 76],
    'Science': [90, 80, 85, 87, 70],
    'English': [82, 75, 89, 84, 77]
})

# Step 2: Merge datasets (Data Integration)
merged_data = pd.merge(student_details, marks_details, on='Student_ID')
print("\nMerged Data:\n", merged_data)

# Step 3: Select features for dimensionality reduction
features = ['Maths', 'Science', 'English']
X = merged_data[features]

# Step 4: Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Apply PCA (reduce to 2 components)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 6: Combine reduced data
reduced_data = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
final_data = pd.concat([merged_data[['Student_ID', 'Name', 'Department']], reduced_data], axis=1)
print("\nReduced Dataset after PCA:\n", final_data)

# Step 7: Visualization
plt.figure(figsize=(7,5))
plt.scatter(final_data['PC1'], final_data['PC2'], c='blue', s=70)
for i, txt in enumerate(final_data['Name']):
    plt.annotate(txt, (final_data['PC1'][i]+0.05, final_data['PC2'][i]+0.05))
plt.title('Visualization of PCA Reduced Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# Step 8: Explained variance
print("\nExplained Variance by each Principal Component:")
print(pca.explained_variance_ratio_)
````

---

## **Result**

The datasets were successfully integrated and dimensionality reduction was performed using PCA, reducing the original three numerical columns to two principal components.

---

## **Output**

The program produced:

* A merged dataset
* PCA-transformed dataset with **PC1** and **PC2**
* Scatter plot of PCA output
* Explained variance ratios for each component

---

## **Conclusion**

This experiment successfully demonstrated:

* Data integration using merging
* Dimensionality reduction using PCA
  The reduced dataset retained essential relationships while simplifying the original dataset structure, enabling efficient analysis and visualization.
