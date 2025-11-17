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
