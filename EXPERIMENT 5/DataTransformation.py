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
