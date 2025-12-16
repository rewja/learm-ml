# 🎯 PANDUAN: TULIS KODE SENDIRI DARI FUNDAMENTAL

Panduan ini akan membimbingmu menulis kode ML dari awal, bukan copy-paste.
Kamu akan paham **kenapa** setiap baris kode penting.

---

## 🧠 MINDSET SEBELUM MULAI

Sebelum ngetik, pahami dulu **alur kerja ML**:
1. **Import library** → butuh tools untuk bekerja
2. **Load data** → butuh data untuk belajar
3. **Siapkan data** → pisahkan input (X) dan output (y)
4. **Bagi data** → training (untuk belajar) & testing (untuk uji)
5. **Buat model** → pilih algoritma
6. **Train model** → model belajar dari data training
7. **Prediksi** → model prediksi data testing
8. **Evaluasi** → ukur seberapa baik model

---

## 📝 LANGKAH 1: IMPORT LIBRARY

**Apa yang harus diketik:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
```

**Kenapa perlu:**
- `numpy` → untuk perhitungan matematika
- `pandas` → untuk mengelola data (seperti Excel)
- `matplotlib` → untuk bikin grafik
- `sklearn` → library ML yang sudah jadi (scikit-learn)
  - `train_test_split` → untuk bagi data
  - `LinearRegression` → model yang mau dipakai
  - `mean_squared_error`, `r2_score` → untuk evaluasi
  - `fetch_california_housing` → dataset yang mau dipakai

**Cara ngetik:**
1. Buat cell baru di Colab
2. Ketik satu per satu import-nya
3. Run cell (Shift + Enter)

**Tips:** Kalau error "ModuleNotFoundError", tambahkan cell ini dulu:
```python
!pip install scikit-learn pandas matplotlib numpy
```

---

## 📝 LANGKAH 2: LOAD DATASET

**Apa yang harus diketik:**
```python
housing = fetch_california_housing()
```

**Kenapa perlu:**
- Ini adalah dataset tentang rumah di California
- Berisi informasi rumah (luas, umur, lokasi, dll)
- Target: prediksi harga rumah

**Lanjutkan dengan:**
```python
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Harga_Rumah'] = housing.target
```

**Penjelasan:**
- `housing.data` → data fitur (input)
- `housing.feature_names` → nama-nama kolom
- `housing.target` → harga rumah (output yang mau diprediksi)
- `pd.DataFrame()` → ubah jadi tabel yang mudah dibaca

**Cek data:**
```python
print(f"Jumlah data: {len(df)}")
df.head()
```

**Yang harus kamu pahami:**
- Dataset punya **fitur** (input) dan **target** (output)
- Fitur = informasi tentang rumah
- Target = harga rumah yang mau diprediksi

---

## 📝 LANGKAH 3: PISAHKAN FITUR DAN TARGET

**Apa yang harus diketik:**
```python
X = df.drop('Harga_Rumah', axis=1)
y = df['Harga_Rumah']
```

**Kenapa perlu:**
- `X` = fitur (input) → semua kolom KECUALI 'Harga_Rumah'
- `y` = target (output) → hanya kolom 'Harga_Rumah'
- Model butuh dipisahkan: input terpisah dari output

**Cek bentuk data:**
```python
print(f"X: {X.shape}")
print(f"y: {y.shape}")
```

**Yang harus kamu pahami:**
- `X.shape` = (jumlah_data, jumlah_fitur)
- `y.shape` = (jumlah_data,)
- Model akan belajar: dari X → prediksi y

---

## 📝 LANGKAH 4: BAGI DATA TRAINING & TESTING

**Apa yang harus diketik:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,
    random_state=42
)
```

**Kenapa perlu:**
- **Training** (80%) → untuk model belajar
- **Testing** (20%) → untuk uji seberapa baik model
- `test_size=0.2` → 20% untuk testing
- `random_state=42` → agar hasilnya konsisten setiap kali run

**Cek pembagian:**
```python
print(f"Training: {X_train.shape[0]} data")
print(f"Testing: {X_test.shape[0]} data")
```

**Yang harus kamu pahami:**
- Model **hanya belajar** dari data training
- Data testing **tidak pernah dilihat** model saat training
- Ini penting untuk tahu apakah model bisa prediksi data baru

---

## 📝 LANGKAH 5: BUAT MODEL

**Apa yang harus diketik:**
```python
model = LinearRegression()
```

**Kenapa perlu:**
- `LinearRegression()` → model yang mencari garis lurus terbaik
- Garis ini menghubungkan fitur (X) dengan harga (y)
- Rumus: `harga = a1*fitur1 + a2*fitur2 + ... + b`

**Yang harus kamu pahami:**
- Model ini masih "kosong", belum belajar apa-apa
- Langkah berikutnya: train model agar bisa belajar

---

## 📝 LANGKAH 6: TRAIN MODEL

**Apa yang harus diketik:**
```python
model.fit(X_train, y_train)
```

**Kenapa perlu:**
- `fit()` → proses model belajar dari data training
- Model mencari pola: fitur → harga
- Proses ini menghitung koefisien (a1, a2, ..., b) yang terbaik

**Cek apa yang dipelajari:**
```python
print("Koefisien:")
print(model.coef_)
print(f"Intercept: {model.intercept_}")
```

**Yang harus kamu pahami:**
- Setelah `fit()`, model sudah "pintar"
- Model sudah tahu pola dari data training
- Sekarang bisa dipakai untuk prediksi

---

## 📝 LANGKAH 7: PREDIKSI

**Apa yang harus diketik:**
```python
y_pred = model.predict(X_test)
```

**Kenapa perlu:**
- `predict()` → model prediksi harga dari fitur
- `X_test` → data testing yang belum pernah dilihat model
- `y_pred` → hasil prediksi model

**Lihat hasil prediksi:**
```python
print("Contoh prediksi:")
for i in range(5):
    print(f"Harga asli: {y_test.iloc[i]:.2f}, Prediksi: {y_pred[i]:.2f}")
```

**Yang harus kamu pahami:**
- Model menggunakan pola yang dipelajari untuk prediksi
- Bandingkan `y_test` (harga asli) dengan `y_pred` (prediksi)
- Semakin dekat, semakin baik model

---

## 📝 LANGKAH 8: EVALUASI MODEL

**Apa yang harus diketik:**
```python
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")
```

**Kenapa perlu:**
- **MSE** (Mean Squared Error) → rata-rata kuadrat error
  - Semakin kecil = semakin baik
  - Mengukur seberapa jauh prediksi dari nilai asli
  
- **R² Score** → seberapa baik model menjelaskan data
  - 0.0 = tidak lebih baik dari rata-rata
  - 1.0 = sempurna
  - Semakin mendekati 1 = semakin baik

**Yang harus kamu pahami:**
- Evaluasi penting untuk tahu kualitas model
- MSE kecil + R² mendekati 1 = model bagus

---

## 📝 LANGKAH 9: VISUALISASI

**Apa yang harus diketik:**
```python
plt.figure(figsize=(10, 5))

# Grafik 1: Prediksi vs Nilai Asli
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2)
plt.xlabel('Harga Asli')
plt.ylabel('Prediksi')
plt.title('Prediksi vs Nilai Asli')
plt.grid(True)

# Grafik 2: Error
plt.subplot(1, 2, 2)
error = y_test - y_pred
plt.hist(error, bins=30)
plt.xlabel('Error')
plt.ylabel('Frekuensi')
plt.title('Distribusi Error')

plt.tight_layout()
plt.show()
```

**Kenapa perlu:**
- Grafik membantu **melihat** seberapa baik model
- Grafik 1: titik-titik yang dekat dengan garis merah = prediksi bagus
- Grafik 2: error yang terpusat di 0 = model bagus

**Yang harus kamu pahami:**
- Visualisasi membantu memahami hasil
- Lebih mudah lihat grafik daripada angka saja

---

## 📝 LANGKAH 10: COBA PREDIKSI DATA BARU

**Apa yang harus diketik:**
```python
# Contoh: rumah baru
rumah_baru = np.array([[
    3.0,    # Median Income
    30.0,   # House Age
    5.0,    # Average Rooms
    1.0,    # Average Bedrooms
    1500.0, # Population
    3.0,    # Average Occupancy
    34.0,   # Latitude
    -118.0  # Longitude
]])

harga_prediksi = model.predict(rumah_baru)[0]
print(f"Prediksi harga: {harga_prediksi:.2f}")
```

**Kenapa perlu:**
- Ini adalah **aplikasi nyata** model
- Model bisa prediksi harga rumah baru yang belum pernah dilihat
- Ini tujuan akhir ML: bisa prediksi data baru

**Yang harus kamu pahami:**
- Model sudah siap dipakai untuk prediksi data baru
- Input: fitur rumah → Output: prediksi harga

---

## ✅ CHECKLIST: APAKAH KAMU SUDAH PAHAM?

Setelah menulis semua kode, tanya ke diri sendiri:

- [ ] **Apa itu X dan y?** → X = input, y = output
- [ ] **Kenapa harus bagi training & testing?** → Training untuk belajar, testing untuk uji
- [ ] **Apa yang terjadi saat `model.fit()`?** → Model belajar pola dari data
- [ ] **Apa bedanya `fit()` dan `predict()`?** → fit() = belajar, predict() = prediksi
- [ ] **Apa arti MSE dan R² Score?** → MSE = error, R² = seberapa baik model
- [ ] **Kenapa perlu visualisasi?** → Untuk lihat hasil dengan mata

---

## 🎯 TIPS MENULIS KODE

1. **Jangan terburu-buru** → ketik satu baris, run, lihat hasil
2. **Baca error message** → biasanya jelas apa yang salah
3. **Coba ubah-ubah nilai** → misalnya `test_size=0.3` atau `test_size=0.1`
4. **Tanya "kenapa?"** → setiap baris kode punya alasan
5. **Jangan takut salah** → error adalah bagian dari belajar

---

## 🆘 JIKA ERROR

### "ModuleNotFoundError"
```python
!pip install scikit-learn pandas matplotlib numpy
```

### "NameError: name 'X' is not defined"
→ Pastikan sudah run cell yang membuat X

### Grafik tidak muncul
→ Pastikan sudah `plt.show()`

### Hasil berbeda setiap run
→ Pastikan pakai `random_state=42` di `train_test_split`

---

## 🎉 SETELAH SELESAI

Kamu sudah:
1. ✅ Menulis kode ML dari awal
2. ✅ Paham setiap langkah
3. ✅ Bisa membuat model sendiri
4. ✅ Bisa evaluasi dan visualisasi

**Selamat! Kamu sudah paham fundamental ML! 🚀**

