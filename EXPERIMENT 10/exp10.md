# **Experiment 10: Compute Point Estimates and Confidence Intervals for Means and Proportions**

## **Aim**
To calculate point estimates and confidence intervals (CIs) for sample means and proportions using t-distribution, z-distribution, and bootstrap methods.

---

## **Apparatus / Tools Required**
- Python (Jupyter Notebook / VS Code)
- Libraries:
  - `numpy`
  - `pandas`
  - `scipy.stats`
  - `statsmodels`
  - `matplotlib`

---

## **Theory**

### **Point Estimate**
A single numerical value that estimates a population parameter (e.g., mean, proportion).

### **Confidence Interval (CI)**
A range of values that likely contains the true population parameter.

### **Types of CI Methods**

#### **1. t-distribution**
- Used when population standard deviation is unknown  
- Typically for small sample sizes (n < 30)

#### **2. z-distribution**
- Used when population standard deviation is known or sample size is large

#### **3. Bootstrap CI**
- Non-parametric  
- Based on repeated resampling with replacement  
- Useful when assumptions of normality are not guaranteed

These methods help measure uncertainty in the point estimates.

---

## **Steps / Procedure**

### **1. Import Required Libraries**
Load numpy, pandas, scipy.stats, statsmodels, matplotlib.

### **2. Create Sample Dataset**
Generate random marks data from a normal distribution.

### **3. Compute Sample Statistics**
Calculate:
- Sample mean  
- Sample variance  
- Sample standard deviation  

### **4. Compute 95% CI for Mean Using t-distribution**

### **5. Compute 95% CI for Mean Using z-distribution**

### **6. Compute CI for a Sample Proportion**
Using z-statistics.

### **7. Generate Bootstrap CI**
Resample data 1000 times to compute empirical confidence interval.

### **8. Visualize Bootstrap Mean Distribution**
Plot histogram with CI bounds.

---

## **Program / Code**

```python
# EXPERIMENT 10: POINT ESTIMATES AND CONFIDENCE INTERVALS

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.stats.api as sms
import matplotlib.pyplot as plt

# Step 1: Create sample dataset
np.random.seed(42)
data = np.random.normal(loc=70, scale=10, size=30)  # mean=70, std=10, n=30
df = pd.DataFrame(data, columns=['Marks'])
print("Sample Data (first 10 rows):\n", df.head(10))

# Step 2: Compute sample statistics
sample_mean = np.mean(data)
sample_std = np.std(data, ddof=1)
n = len(data)
print("\nSample Mean:", round(sample_mean, 3))
print("Sample Standard Deviation:", round(sample_std, 3))
print("Sample Size:", n)

# Step 3: Confidence Interval using t-distribution
confidence_level = 0.95
alpha = 1 - confidence_level
t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
margin_of_error = t_critical * (sample_std / np.sqrt(n))
t_conf_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)
print("\n95% CI for Mean (t-distribution):", t_conf_interval)

# Step 4: Confidence Interval using z-distribution
z_critical = stats.norm.ppf(1 - alpha/2)
z_margin = z_critical * (sample_std / np.sqrt(n))
z_conf_interval = (sample_mean - z_margin, sample_mean + z_margin)
print("\n95% CI for Mean (z-distribution):", z_conf_interval)

# Step 5: Confidence Interval for Proportion
x = 40  # successes
n_p = 100  # total
p_hat = x / n_p
se = np.sqrt((p_hat * (1 - p_hat)) / n_p)
z_critical = stats.norm.ppf(1 - alpha/2)
prop_ci = (p_hat - z_critical * se, p_hat + z_critical * se)
print("\n95% CI for Proportion:", prop_ci)

# Step 6: Bootstrap Confidence Interval
bootstrap_means = []
for i in range(1000):
    sample = np.random.choice(data, size=n, replace=True)
    bootstrap_means.append(np.mean(sample))
lower_bound = np.percentile(bootstrap_means, 2.5)
upper_bound = np.percentile(bootstrap_means, 97.5)
bootstrap_ci = (lower_bound, upper_bound)
print("\n95% Bootstrap CI for Mean:", bootstrap_ci)

# Step 7: Visualization
plt.figure(figsize=(8,5))
plt.hist(bootstrap_means, bins=25, color='lightblue', edgecolor='black')
plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound (2.5%)')
plt.axvline(upper_bound, color='green', linestyle='--', label='Upper Bound (97.5%)')
plt.title('Bootstrap Distribution of Sample Mean')
plt.xlabel('Mean Values')
plt.ylabel('Frequency')
plt.legend()
plt.show()
````

---

## **Result**

* Sample mean, variance, and standard deviation were computed.
* Confidence intervals were successfully calculated using:

  * t-distribution
  * z-distribution
  * bootstrap resampling
* CI for a sample proportion was also computed.

---

## **Output**

The program displayed:

* Sample mean and standard deviation
* 95% CI using t and z statistics
* CI for sample proportion
* Bootstrap CI with histogram and CI lines

---

## **Conclusion**

This experiment demonstrated how to compute point estimates and confidence intervals using analytical (t/z) and empirical (bootstrap) approaches.
All methods produced consistent estimates, confirming their usefulness in statistical inference.

