import streamlit as st
import pandas as pd
import plotly.express as px

# Load your data
df = pd.read_csv('sample_data.csv')  # replace with your CSV file

# Sidebar filters
product_filter = st.sidebar.multiselect('Select Product', options=df['Product'].unique())
time_filter = st.sidebar.date_input('Select Date Range', [])

# Apply filters
if product_filter:
    filtered_df = df[df['Product'].isin(product_filter)]
else:
    filtered_df = df.copy()  # show all data if no product selected

# Check if filtered data is empty
if filtered_df.empty:
    st.warning("No data available for the selected filters.")
else:
    # KPIs
    total_revenue = filtered_df['Revenue'].sum()
    avg_revenue = filtered_df['Revenue'].mean()
    top_product = filtered_df.groupby('Product')['Revenue'].sum().idxmax()

    st.metric("Total Revenue", f"${total_revenue:,.2f}")
    st.metric("Average Revenue", f"${avg_revenue:,.2f}")
    st.metric("Top Product", top_product)

    # Chart
    fig = px.bar(
        filtered_df.groupby('Product')['Revenue'].sum().reset_index(),
        x='Product',
        y='Revenue',
        title='Revenue by Product'
    )
    st.plotly_chart(fig)
