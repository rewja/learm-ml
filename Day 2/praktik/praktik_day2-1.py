import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Harga_Rumah'] = housing.target

print(f"Jumlah data: {len(df)}")
df.head()

x = df.drop('Harga_Rumah', axis=1)
y = df['Harga_Rumah']

print(f"x: {x.shape}")
print(f"y: {y.shape}")

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.2,
    random_state = 42
)

print(f"Training: {x_train.shape[0]} data")
print(f"Testing: {x_test.shape[0]} data")

# Ridge Regression 
# ridge = Pipeline([
#     ('scaler', StandardScaler()),
#     ('ridge', Ridge(alpha=100))
# ])

# ridge.fit(x_train, y_train)
# y_pred_ridge = ridge.predict(x_test)

# mse_ridge = mean_squared_error(y_test, y_pred_ridge)
# r2_ridge = r2_score(y_test, y_pred_ridge)

# print(f"MSE ridge: {mse_ridge:.4f}")
# print(f"R² ridge: {r2_ridge:.4f}")

# Lasso Regression 
# lasso = Pipeline([
#     ('scaler', StandardScaler()),
#     ('lasso', Lasso(alpha=0.001, max_iter=10000))
# ])

# lasso.fit(x_train, y_train)
# y_pred_lasso = lasso.predict(x_test)

for a in [0.0001, 0.001, 0.01]:
    lasso = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', Lasso(alpha=a, max_iter=10000))
    ])
    lasso.fit(x_train, y_train)
    y_pred = lasso.predict(x_test)
    print(f"alpha={a}")
    print("Testing data:")
    for i in range(5):
        print(f"Harga asli: {y_test.iloc[i]:.2f}, Prediksi: {y_pred[i]:.2f}")
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")

# mse_lasso = mean_squared_error(y_test, y_pred_lasso)
# r2_lasso = r2_score(y_test, y_pred_lasso)

# print(f"MSE lasso: {mse_lasso:.4f}")
# print(f"R² lasso: {r2_lasso:.4f}")