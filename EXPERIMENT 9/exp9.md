# **Experiment 9: Case Study – Full Exploratory Data Analysis (EDA)**

## **Aim**
To perform a complete Exploratory Data Analysis (EDA) including data cleaning, statistical summaries, visualization, correlation analysis, and hypothesis formation using the Titanic dataset.

---

## **Apparatus / Tools Required**
- Python (Jupyter Notebook / VS Code)
- Libraries:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
  - `scikit-learn`

---

## **Theory**

Exploratory Data Analysis (EDA) helps understand the underlying structure, trends, and relationships in a dataset before performing machine learning or statistical modeling.

### **Key Components of EDA**
#### **1. Data Cleaning**
- Handling missing values  
- Removing inconsistent entries  

#### **2. Descriptive Statistics**
- Summary of numerical and categorical variables  

#### **3. Univariate Analysis**
Exploring distribution of single variables (e.g., age, fare)

#### **4. Bivariate / Multivariate Analysis**
Studying relationships such as:
- Gender vs Survival  
- Class vs Survival  

#### **5. Visual Storytelling**
Using data visualizations to support patterns, trends, and hypotheses.

---

## **Steps / Procedure**

### **1. Import Libraries**
Load pandas, numpy, matplotlib, seaborn.

### **2. Load Dataset**
Use the Titanic dataset available in seaborn.

### **3. Data Inspection**
Analyze:
- Structure  
- Data types  
- Missing values  

### **4. Data Cleaning**
- Fill missing values for age and embarked  
- Drop deck column due to heavy missingness  

### **5. Summary Statistics**
Generate descriptive statistics for numerical and categorical data.

### **6. Univariate Analysis**
Use:
- Histograms  
- Boxplots  

### **7. Bivariate Analysis**
Explore relationships using:
- Boxplots  
- Barplots  

### **8. Correlation Analysis**
Compute correlations for numeric columns and display heatmap.

### **9. Hypothesis Testing**
Example hypothesis:
> Women had higher survival rate than men.

### **10. Visual Storytelling**
Use layered plots to support insights.

---

## **Program / Code**

```python
# EXPERIMENT 9: FULL EDA - TITANIC CASE STUDY

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Step 1: Load dataset
data = sns.load_dataset('titanic')
print("Dataset Loaded Successfully.\n")
print("Initial Data Overview:")
print(data.head())

# Step 2: Data Summary
print("\nData Information:")
print(data.info())
print("\nMissing Values:")
print(data.isnull().sum())

# Step 3: Handle missing values
data['age'].fillna(data['age'].median(), inplace=True)
data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)
data.drop(columns=['deck'], inplace=True)

# Step 4: Summary Statistics
print("\nDescriptive Statistics:")
print(data.describe(include='all'))

# Step 5: Univariate Analysis
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(data['age'], kde=True, color='skyblue')
plt.title("Age Distribution")
plt.subplot(1, 2, 2)
sns.boxplot(x='fare', data=data, color='lightgreen')
plt.title("Fare Distribution with Outliers")
plt.tight_layout()
plt.show()

# Step 6: Bivariate Analysis
plt.figure(figsize=(6,5))
sns.boxplot(x='sex', y='age', data=data, palette='pastel')
plt.title("Age Distribution by Gender")
plt.show()

plt.figure(figsize=(6,5))
sns.barplot(x='sex', y='survived', data=data, palette='coolwarm')
plt.title("Survival Rate by Gender")
plt.show()

# Step 7: Correlation Analysis
numeric_data = data.select_dtypes(include=[np.number])
corr = numeric_data.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='YlGnBu')
plt.title("Correlation Heatmap")
plt.show()

# Step 8: Hypothesis Testing
print("\nHypothesis: Women had higher survival rate than men.")
survival_by_gender = data.groupby('sex')['survived'].mean()
print("Average Survival Rate by Gender:\n", survival_by_gender)

# Step 9: Visual Storytelling
plt.figure(figsize=(8,5))
sns.histplot(data=data, x='age', hue='survived', multiple='stack', palette='Set2')
plt.title("Age vs Survival Distribution")
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x='class', y='survived', hue='sex', data=data, palette='mako')
plt.title("Survival Rate by Passenger Class and Gender")
plt.show()

# Step 10: Summary of Findings
print("\n--- SUMMARY OF INSIGHTS ---")
print("1. Average passenger age is around", round(data['age'].mean(), 2))
print("2. Younger passengers tended to survive more often.")
print("3. Females had significantly higher survival rates than males.")
print("4. Higher class passengers had better survival chances.")
print("5. Fare positively correlates with survival probability.")
````

---

## **Result**

EDA was successfully performed on the Titanic dataset. The dataset was cleaned, analyzed, and visualized to uncover patterns and relationships.

---

## **Output**

The analysis generated:

* Summary statistics
* Histograms, boxplots, and heatmaps
* Survival trends by gender, age, and class
* Hypothesis validation showing women had higher survival rates
* Visual storytelling using multi-variable plots

---

## **Conclusion**

The experiment demonstrated comprehensive EDA including data cleaning, visualization, correlation analysis, and hypothesis testing.
Insights revealed how demographic and socio-economic factors influenced Titanic survival patterns.

