import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import GridSearchCV

housing = fetch_california_housing()

df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Harga_Rumah'] = housing.target

print(f"Jumlah data: {len(df)}")
df.head()

x = df.drop('Harga_Rumah', axis=1)
y = df['Harga_Rumah']

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size = 0.2,
    random_state = 42
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(max_iter=10000))
])

param_grid = {
    'lasso__alpha': [0.0001, 0.001, 0.01, 0.05, 0.1]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv = 5,
    scoring = 'r2',
)

grid.fit(x_train, y_train)

best_model = grid.best_estimator_
y_pred = best_model.predict(x_test)

print("Testing data:")
for i in range(5):
    print(f"Harga asli: {y_test.iloc[i]:.2f}, Prediksi: {y_pred[i]:.2f}")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
