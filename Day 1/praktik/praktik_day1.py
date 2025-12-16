import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

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

scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

print(f"Training: {x_train_scaled.shape[0]} data")
print(f"Testing: {x_test_scaled.shape[0]} data")

model = LinearRegression()

model.fit(x_train_scaled, y_train)

print("Koefisien:")
print(model.coef_)
print(f"Intercept: {model.intercept_}")

y_pred = model.predict(x_test_scaled)

print("Contoh prediksi:")
for i in range(5):
  print(f"Harga asli: {y_test.iloc[i]:.2f}, Prediksi: {y_pred[i]:.2f}")

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:4f}")
print(f"R2: {r2:4f}")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2)
plt.xlabel('Harga Asli')
plt.ylabel('Prediksi')
plt.title('Prediksi vs Nilai Asli')
plt.grid(True)


plt.subplot(1, 2, 2)
error = y_test - y_pred
plt.hist(error, bins=30)
plt.xlabel('Error')
plt.ylabel('Frekuensi')
plt.title('Distribusi Error')

plt.tight_layout()
plt.show()

rumah_baru = np.array([[ 
    3.0,   # MedInc
    20.0,  # HouseAge
    5.0,   # AveRooms
    1.0,   # AveBedrms
    1500,  # Population
    3.0,   # AveOccup
    34.2,  # Latitude
    -118.4 # Longitude
]])

rumah_baru_scaled = scaler.transform(rumah_baru)
harga_prediksi = model.predict(rumah_baru_prediksi)[0]
print(f"Prdiksi harga: {harga_prediksi:.2f}")