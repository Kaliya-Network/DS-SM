# **Experiment 8: Perform Feature Selection by Ranking or Feature Importance**

## **Aim**
To perform feature selection using ranking and feature importance methods such as SelectKBest and Recursive Feature Elimination (RFE), and to implement a pipeline for combined preprocessing and model evaluation.

---

## **Apparatus / Tools Required**
- Python (Jupyter Notebook / VS Code)
- Libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`

---

## **Theory**

Feature selection is used to reduce dimensionality by identifying features that contribute the most to model performance while improving interpretability.

### **SelectKBest**
A *filter-based* method that selects top features using statistical scores such as:
- **ANOVA F-test**
- Mutual information

### **RFE (Recursive Feature Elimination)**
A *wrapper-based* method that:
- Trains a model  
- Ranks features based on coefficients  
- Iteratively removes less important features  

### **Pipelines**
Automate multiple steps such as:
- Scaling  
- Feature selection  
- Model training  
in a single workflow.

---

## **Steps / Procedure**

### **1. Import Libraries**
Use pandas, numpy, and scikit-learn modules.

### **2. Load Sample Dataset**
Load the Breast Cancer dataset from scikit-learn.

### **3. Split Dataset**
Use `train_test_split()` to divide data into training and testing sets.

### **4. Apply SelectKBest**
Select top k features using:
- `SelectKBest(score_func=f_classif, k=5)`

### **5. Apply RFE**
Use Logistic Regression with:
- `RFE(estimator, n_features_to_select=5)`

### **6. Build Pipeline**
Pipeline steps:
- Standardization  
- SelectKBest  
- Logistic Regression  

### **7. Train and Evaluate Pipeline**
Calculate model accuracy on test data.

### **8. Compare Feature Rankings**
Display ranked feature scores and RFE rankings.

---

## **Program / Code**

```python
# EXPERIMENT 8: FEATURE SELECTION USING SELECTKBEST AND RFE

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Step 1: Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("Dataset Shape:", X.shape)
print("Target Classes:", data.target_names)

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Feature Selection using SelectKBest
selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X_train, y_train)
selected_features = X_train.columns[selector.get_support()]
print("\nTop 5 Selected Features (SelectKBest):")
print(selected_features.tolist())

# Step 4: Feature Selection using RFE
model = LogisticRegression(max_iter=1000)
rfe = RFE(model, n_features_to_select=5)
rfe.fit(X_train, y_train)
rfe_features = X_train.columns[rfe.support_]
print("\nTop 5 Selected Features (RFE):")
print(rfe_features.tolist())

# Step 5: Pipeline (Scaling + Feature Selection + Classification)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_classif, k=5)),
    ('classifier', LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
accuracy = pipeline.score(X_test, y_test)
print("\nPipeline Accuracy: {:.2f}%".format(accuracy * 100))

# Step 6: Feature Ranking Comparison
ranking = pd.DataFrame({
    'Feature': X_train.columns,
    'RFE_Ranking': rfe.ranking_,
    'SelectKBest_Score': selector.scores_
}).sort_values(by='SelectKBest_Score', ascending=False)

print("\nFeature Ranking Comparison:\n", ranking.head(10))
````

---

## **Result**

* Top features were selected using both SelectKBest and RFE.
* A combined pipeline executed scaling, feature selection, and classification efficiently.
* The pipeline produced an accuracy score on the test dataset.

---

## **Output**

The program displayed:

* Top 5 features selected by SelectKBest
* Top 5 features selected by RFE
* Pipeline accuracy percentage
* A feature ranking comparison table

---

## **Conclusion**

This experiment successfully demonstrated feature selection using both filter-based (SelectKBest) and wrapper-based (RFE) techniques.
These methods reduce dataset complexity while improving model performance and interpretability.
