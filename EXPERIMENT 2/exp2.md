# **Experiment 2: Create a Simple Dashboard using Streamlit, Pandas, and Plotly**

## **Aim**
To create an interactive dashboard using Streamlit, Pandas, and Plotly that can analyze tabular data and display key business or domain metrics such as revenue, top products, or performance over time.

---

## **Apparatus / Tools Required**
- Python (Jupyter / VS Code / Anaconda)
- Libraries:
  - `pandas`
  - `streamlit`
  - `plotly`
- Dataset: CSV or tabular data (e.g., **sales_data.csv**)

---

## **Theory**

Dashboards are interactive tools used to visualize and analyze data dynamically.

In Python:

- **Streamlit** is used for creating data-driven web applications without complex front-end development.
- **Pandas** handles data loading, cleaning, filtering, and aggregation.
- **Plotly** provides interactive visualizations such as bar charts, line graphs, and more.

This experiment demonstrates how to create a simple dashboard for business, medical, or banking datasets using filters, KPIs, and interactive charts.

---

## **Steps / Procedure**

### **Step 1: Prepare the Environment**
Install required Python libraries:

```

pip install pandas streamlit plotly

```

---

### **Step 2: Load and Process the Data**
- Load a tabular dataset using `pd.read_csv()`.
- Perform aggregations such as sales by product or by time.
- Calculate KPIs like:
  - Total revenue  
  - Average sales  
  - Top-performing product  

---

### **Step 3: Build Dashboard Interface Using Streamlit**
- Create sidebar filters such as:
  - Product selection
  - Date range
- Display KPIs using `st.metric()`
- Plot interactive charts using Plotly Express (`px.bar`, `px.line`, etc.)
- Display filtered data using `st.dataframe()`

---

### **Step 4: Run the Dashboard**
Save the file as **dashboard.py** and run:

```

streamlit run dashboard.py

````

This will open the dashboard in the browser with all interactive components.

---

## **Program / Code**

```python
import streamlit as st
import pandas as pd
import plotly.express as px

# Load your data
df = pd.read_csv('sales_data.csv')  # replace with your CSV file

# Create filters in sidebar
product_filter = st.sidebar.multiselect('Select Product', options=df['Product'].unique())
time_filter = st.sidebar.date_input('Select Date Range', [])

# Apply filters
filtered_df = df[df['Product'].isin(product_filter)]
# Add date filtering if dates are available in your data

# KPIs
total_revenue = filtered_df['Revenue'].sum()
avg_revenue = filtered_df['Revenue'].mean()
top_product = filtered_df.groupby('Product')['Revenue'].sum().idxmax()

st.metric("Total Revenue", f"${total_revenue:,.2f}")
st.metric("Average Revenue", f"${avg_revenue:,.2f}")
st.metric("Top Product", top_product)

# Charts
fig = px.bar(
    filtered_df.groupby('Product')['Revenue'].sum().reset_index(),
    x='Product',
    y='Revenue',
    title='Revenue by Product'
)
st.plotly_chart(fig)
````

---

## **Result**

A fully functional interactive dashboard was created using Streamlit, Pandas, and Plotly.
The dashboard displays total and average revenue, identifies top-performing products, and allows users to filter and visualize data dynamically.

---

## **Conclusion**

This experiment successfully demonstrates how to integrate data analysis and visualization using Pandas, Streamlit, and Plotly.
Such dashboards are powerful tools for answering real-world business, medical, or banking questions using data-driven insights.

