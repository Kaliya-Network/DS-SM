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
