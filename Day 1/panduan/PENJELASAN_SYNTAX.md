# 📖 PENJELASAN SYNTAX PER BARIS

Penjelasan detail setiap syntax, untuk apa, dan fungsinya.

---

## 📦 IMPORT LIBRARY

### `import numpy as np`
**Apa itu:** Import library NumPy, beri alias `np`

**Fungsi:** 
- NumPy = Numerical Python
- Untuk perhitungan matematika (array, matrix)
- Alias `np` = singkatan agar tidak perlu ketik `numpy` panjang

**Contoh penggunaan:**
```python
arr = np.array([1, 2, 3])  # buat array
hasil = np.mean([1, 2, 3])  # hitung rata-rata
```

**Kenapa perlu:** Model ML butuh perhitungan matematika, NumPy cepat dan efisien

---

### `import pandas as pd`
**Apa itu:** Import library Pandas, beri alias `pd`

**Fungsi:**
- Pandas = untuk mengelola data (seperti Excel di Python)
- Bisa baca, edit, analisis data dalam bentuk tabel
- Alias `pd` = singkatan

**Contoh penggunaan:**
```python
df = pd.DataFrame({'nama': ['A', 'B'], 'umur': [20, 25]})  # buat tabel
df.head()  # lihat 5 baris pertama
df['umur']  # ambil kolom 'umur'
```

**Kenapa perlu:** Dataset ML biasanya dalam bentuk tabel, Pandas memudahkan

---

### `import matplotlib.pyplot as plt`
**Apa itu:** Import bagian plotting dari Matplotlib, beri alias `plt`

**Fungsi:**
- Matplotlib = library untuk bikin grafik
- `pyplot` = bagian yang paling sering dipakai
- Alias `plt` = singkatan

**Contoh penggunaan:**
```python
plt.plot([1, 2, 3], [4, 5, 6])  # bikin garis
plt.scatter([1, 2], [3, 4])  # bikin titik
plt.show()  # tampilkan grafik
```

**Kenapa perlu:** Visualisasi hasil ML penting untuk paham seberapa baik model

**Alternatif Matplotlib:**
Matplotlib punya beberapa cara/cara pakai:

1. **pyplot (yang kita pakai)** - Paling mudah untuk pemula
   ```python
   import matplotlib.pyplot as plt
   plt.plot([1, 2, 3])
   plt.show()
   ```
   - ✅ Mudah dipakai, seperti MATLAB
   - ✅ Cocok untuk grafik sederhana
   - ✅ Paling populer untuk pemula

2. **Object-Oriented API** - Lebih fleksibel, untuk grafik kompleks
   ```python
   import matplotlib.pyplot as plt
   fig, ax = plt.subplots()  # buat figure dan axes
   ax.plot([1, 2, 3])  # plot di axes
   fig.show()  # tampilkan figure
   ```
   - ✅ Lebih eksplisit (jelas mana figure, mana axes)
   - ✅ Lebih mudah untuk multiple subplots
   - ✅ Lebih baik untuk grafik kompleks
   - ❌ Sedikit lebih verbose (lebih panjang)

3. **Pylab** - Tidak disarankan (sudah deprecated)
   ```python
   from pylab import *  # jangan pakai ini!
   ```

**Kenapa pakai pyplot?**
- ✅ **Paling mudah** untuk pemula
- ✅ **Paling populer** - banyak tutorial pakai ini
- ✅ **Cukup untuk grafik sederhana** (seperti yang kita butuhkan)
- ✅ **Lebih singkat** - tidak perlu buat figure/axes manual

**Kapan pakai Object-Oriented API?**
- Saat butuh grafik yang lebih kompleks
- Saat butuh kontrol lebih detail
- Saat bekerja dengan multiple figures

**Untuk sekarang:** pyplot sudah cukup! Fokus belajar ML dulu, visualisasi bisa pakai yang mudah.

---

### `from sklearn.model_selection import train_test_split`
**Apa itu:** Import fungsi `train_test_split` dari scikit-learn

**Fungsi:**
- `train_test_split` = fungsi untuk membagi data menjadi training & testing
- Bagian dari scikit-learn (library ML)

**Contoh penggunaan:**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

**Kenapa perlu:** Harus bagi data sebelum training, fungsi ini otomatis

---

### `from sklearn.linear_model import LinearRegression`
**Apa itu:** Import model Linear Regression dari scikit-learn

**Fungsi:**
- `LinearRegression` = class/model untuk linear regression
- Model ini mencari garis lurus terbaik untuk prediksi

**Contoh penggunaan:**
```python
model = LinearRegression()  # buat model
model.fit(X_train, y_train)  # train model
```

**Kenapa perlu:** Ini adalah model ML yang akan dipakai untuk prediksi

---

### `from sklearn.metrics import mean_squared_error, r2_score`
**Apa itu:** Import fungsi evaluasi dari scikit-learn

**Fungsi:**
- `mean_squared_error` = hitung MSE (Mean Squared Error)
- `r2_score` = hitung R² Score
- Keduanya untuk mengukur seberapa baik model

**Contoh penggunaan:**
```python
mse = mean_squared_error(y_test, y_pred)  # hitung error
r2 = r2_score(y_test, y_pred)  # hitung R²
```

**Kenapa perlu:** Butuh ukur kualitas model, tidak bisa cuma lihat prediksi

---

### `from sklearn.datasets import fetch_california_housing`
**Apa itu:** Import fungsi untuk load dataset California Housing

**Fungsi:**
- `fetch_california_housing` = fungsi untuk download & load dataset
- Dataset berisi informasi rumah di California

**Contoh penggunaan:**
```python
housing = fetch_california_housing()  # load dataset
```

**Kenapa perlu:** Butuh dataset untuk latihan, ini dataset siap pakai

---

## 📊 LOAD DATASET

### `housing = fetch_california_housing()`
**Apa itu:** Panggil fungsi untuk load dataset, simpan ke variabel `housing`

**Fungsi:**
- Download & load dataset California Housing
- Dataset berisi data rumah (fitur) dan harga (target)
- `housing` = object yang berisi data, fitur names, target

**Struktur `housing`:**
- `housing.data` = data fitur (input)
- `housing.feature_names` = nama-nama kolom
- `housing.target` = harga rumah (output)

**Kenapa perlu:** Butuh data untuk latihan model

---

### `df = pd.DataFrame(housing.data, columns=housing.feature_names)`
**Apa itu:** Ubah data menjadi DataFrame (tabel), dengan nama kolom

**Penjelasan per bagian:**
- `pd.DataFrame()` = fungsi untuk buat tabel
- `housing.data` = data yang mau diubah (array)
- `columns=housing.feature_names` = nama kolom dari dataset
- `df` = variabel untuk menyimpan tabel

**Contoh hasil:**
```
   MedInc  HouseAge  AveRooms  ...
0    8.3     41.0      6.9    ...
1    8.3     21.0      6.9    ...
```

**Kenapa perlu:** DataFrame lebih mudah dibaca & dimanipulasi daripada array

---

### `df['Harga_Rumah'] = housing.target`
**Apa itu:** Tambahkan kolom baru 'Harga_Rumah' ke tabel, isinya `housing.target`

**Penjelasan per bagian:**
- `df['Harga_Rumah']` = kolom baru bernama 'Harga_Rumah'
- `=` = assignment (isi nilai)
- `housing.target` = harga rumah dari dataset

**Hasil:** Tabel `df` sekarang punya kolom tambahan 'Harga_Rumah'

**Kenapa perlu:** Target (harga) perlu ada di tabel untuk mudah diakses

---

### `print(f"Jumlah data: {len(df)}")`
**Apa itu:** Print jumlah baris data

**Penjelasan per bagian:**
- `print()` = fungsi untuk tampilkan output
- `f"..."` = f-string (format string), bisa masukkan variabel
- `{len(df)}` = jumlah baris di DataFrame `df`
- `len()` = fungsi untuk hitung panjang/jumlah

**Output contoh:** `Jumlah data: 20640`

**Kenapa perlu:** Cek apakah data sudah ter-load dengan benar

---

### `df.head()`
**Apa itu:** Tampilkan 5 baris pertama dari tabel

**Fungsi:**
- Method dari Pandas DataFrame
- Menampilkan 5 baris pertama (default)
- Berguna untuk melihat preview data

**Kenapa perlu:** Lihat struktur data sebelum lanjut

---

## 🔧 PERSIAPAN DATA

### `X = df.drop('Harga_Rumah', axis=1)`
**Apa itu:** Buat variabel X yang berisi semua kolom KECUALI 'Harga_Rumah'

**Penjelasan per bagian:**
- `df.drop()` = method untuk hapus kolom/baris
- `'Harga_Rumah'` = kolom yang mau dihapus
- `axis=1` = artinya hapus kolom (bukan baris)
  - `axis=0` = baris
  - `axis=1` = kolom
- `X` = variabel untuk menyimpan fitur (input)

**Hasil:** X berisi semua fitur (MedInc, HouseAge, dll) tanpa Harga_Rumah

**Kenapa perlu:** Model butuh input (X) terpisah dari output (y)

---

### `y = df['Harga_Rumah']`
**Apa itu:** Ambil kolom 'Harga_Rumah' saja, simpan ke variabel y

**Penjelasan per bagian:**
- `df['Harga_Rumah']` = akses kolom 'Harga_Rumah' dari DataFrame
- `y` = variabel untuk menyimpan target (output)

**Hasil:** y berisi hanya harga rumah

**Kenapa perlu:** Model butuh target (y) terpisah untuk belajar

---

### `print(f"X: {X.shape}")`
**Apa itu:** Print bentuk/dimensi dari X

**Penjelasan per bagian:**
- `X.shape` = atribut yang berisi dimensi (baris, kolom)
- Contoh: `(20640, 8)` = 20640 baris, 8 kolom
- `f"..."` = f-string untuk format output

**Output contoh:** `X: (20640, 8)`

**Kenapa perlu:** Cek apakah X sudah benar (punya fitur, tanpa target)

---

## ✂️ SPLIT DATA

### `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)`
**Apa itu:** Bagi data menjadi 4 bagian: training & testing untuk X dan y

**Penjelasan per bagian:**
- `train_test_split()` = fungsi untuk bagi data
- `X, y` = data yang mau dibagi
- `test_size=0.2` = 20% untuk testing, 80% untuk training
- `random_state=42` = seed untuk random, agar hasil konsisten
- `X_train, X_test` = fitur untuk training & testing
- `y_train, y_test` = target untuk training & testing

**Hasil:** 4 variabel baru:
- `X_train` = fitur untuk training
- `X_test` = fitur untuk testing
- `y_train` = target untuk training
- `y_test` = target untuk testing

**Kenapa perlu:** Model harus belajar dari training, diuji dengan testing

---

## 🤖 BUAT MODEL

### `model = LinearRegression()`
**Apa itu:** Buat instance/object model Linear Regression

**Penjelasan per bagian:**
- `LinearRegression()` = constructor untuk buat model baru
- `model` = variabel untuk menyimpan model
- Model ini masih "kosong", belum belajar

**Fungsi model:**
- Mencari garis lurus terbaik: `y = a1*x1 + a2*x2 + ... + b`
- `a1, a2, ...` = koefisien (weight)
- `b` = intercept (bias)

**Kenapa perlu:** Butuh model untuk belajar dari data

---

## 🏋️ TRAIN MODEL

### `model.fit(X_train, y_train)`
**Apa itu:** Latih model dengan data training

**Penjelasan per bagian:**
- `model.fit()` = method untuk train model
- `X_train` = fitur untuk training (input)
- `y_train` = target untuk training (output yang benar)
- Proses ini menghitung koefisien terbaik

**Apa yang terjadi:**
1. Model melihat pola: fitur → target
2. Menghitung koefisien (a1, a2, ..., b) yang terbaik
3. Menyimpan koefisien di dalam model

**Setelah `fit()`:**
- Model sudah "pintar"
- Bisa dipakai untuk prediksi
- Koefisien tersimpan di `model.coef_` dan `model.intercept_`

**Kenapa perlu:** Model harus belajar dulu sebelum bisa prediksi

---

### `model.coef_`
**Apa itu:** Atribut yang berisi koefisien yang dipelajari model

**Fungsi:**
- Array berisi koefisien untuk setiap fitur
- Contoh: `[0.4, -0.1, 0.3, ...]`
- Setiap angka = seberapa penting fitur tersebut

**Kenapa perlu:** Lihat apa yang dipelajari model

---

### `model.intercept_`
**Apa itu:** Atribut yang berisi intercept (bias) yang dipelajari model

**Fungsi:**
- Nilai konstan dalam persamaan linear
- Contoh: `2.5`
- Bagian `b` dalam `y = a1*x1 + ... + b`

**Kenapa perlu:** Lihat bagian konstan dari model

---

## 🔮 PREDIKSI

### `y_pred = model.predict(X_test)`
**Apa itu:** Prediksi target dari fitur testing

**Penjelasan per bagian:**
- `model.predict()` = method untuk prediksi
- `X_test` = fitur testing (data baru yang belum pernah dilihat)
- `y_pred` = hasil prediksi (array)

**Apa yang terjadi:**
1. Model menggunakan koefisien yang sudah dipelajari
2. Menghitung: `prediksi = a1*x1 + a2*x2 + ... + b`
3. Mengembalikan array prediksi

**Hasil:** Array prediksi dengan panjang sama seperti `y_test`

**Kenapa perlu:** Ini tujuan akhir ML - bisa prediksi data baru

---

### `y_test.iloc[i]`
**Apa itu:** Akses elemen ke-i dari Series y_test

**Penjelasan per bagian:**
- `y_test` = Series (array dengan index)
- `.iloc[i]` = integer location, akses berdasarkan posisi
- `i` = index (0, 1, 2, ...)

**Contoh:**
```python
y_test.iloc[0]  # elemen pertama
y_test.iloc[1]  # elemen kedua
```

**Kenapa perlu:** Untuk akses satu per satu saat loop

---

### `y_pred[i]`
**Apa itu:** Akses elemen ke-i dari array y_pred

**Penjelasan per bagian:**
- `y_pred` = array NumPy
- `[i]` = indexing, akses berdasarkan posisi
- `i` = index (0, 1, 2, ...)

**Kenapa perlu:** Untuk akses satu per satu saat loop

---

### `abs(y_test.iloc[i] - y_pred[i])`
**Apa itu:** Hitung selisih absolut antara nilai asli dan prediksi

**Penjelasan per bagian:**
- `y_test.iloc[i]` = nilai asli
- `y_pred[i]` = prediksi
- `-` = pengurangan
- `abs()` = fungsi untuk nilai absolut (selalu positif)

**Hasil:** Selisih antara prediksi dan nilai asli

**Kenapa perlu:** Lihat seberapa jauh prediksi dari nilai sebenarnya

---

## 📈 EVALUASI

### `mse = mean_squared_error(y_test, y_pred)`
**Apa itu:** Hitung Mean Squared Error (MSE)

**Penjelasan per bagian:**
- `mean_squared_error()` = fungsi untuk hitung MSE
- `y_test` = nilai asli (ground truth)
- `y_pred` = prediksi model
- `mse` = variabel untuk menyimpan hasil

**Rumus MSE:**
```
MSE = rata-rata((y_test - y_pred)²)
```

**Interpretasi:**
- Semakin kecil = semakin baik
- 0 = sempurna (tidak ada error)

**Kenapa perlu:** Ukur kualitas model secara numerik

---

### `r2 = r2_score(y_test, y_pred)`
**Apa itu:** Hitung R² Score (R-squared)

**Penjelasan per bagian:**
- `r2_score()` = fungsi untuk hitung R²
- `y_test` = nilai asli
- `y_pred` = prediksi model
- `r2` = variabel untuk menyimpan hasil

**Interpretasi:**
- 0.0 = model tidak lebih baik dari rata-rata
- 1.0 = sempurna (prediksi 100% akurat)
- Bisa negatif = model sangat buruk

**Kenapa perlu:** Ukur seberapa baik model menjelaskan variasi data

---

### `f"MSE: {mse:.4f}"`
**Apa itu:** Format string dengan 4 angka di belakang koma

**Penjelasan per bagian:**
- `f"..."` = f-string
- `{mse}` = masukkan variabel mse
- `:.4f` = format: 4 angka di belakang koma (float)

**Contoh output:** `MSE: 0.5552`

**Kenapa perlu:** Tampilkan hasil dengan format yang rapi

---

## 📊 VISUALISASI

### `plt.figure(figsize=(10, 5))`
**Apa itu:** Buat figure (canvas) untuk grafik dengan ukuran tertentu

**Penjelasan per bagian:**
- `plt.figure()` = fungsi untuk buat figure baru
- `figsize=(10, 5)` = ukuran: lebar 10, tinggi 5 (dalam inch)

**Kenapa perlu:** Atur ukuran grafik agar tidak terlalu kecil

---

### `plt.subplot(1, 2, 1)`
**Apa itu:** Buat subplot (grafik kecil) di posisi tertentu

**Penjelasan per bagian:**
- `plt.subplot()` = fungsi untuk buat subplot
- `1, 2, 1` = 1 baris, 2 kolom, subplot ke-1
- Artinya: buat 2 grafik sejajar, ini yang pertama

**Kenapa perlu:** Bisa tampilkan beberapa grafik dalam satu figure

---

### `plt.scatter(y_test, y_pred, alpha=0.5)`
**Apa itu:** Buat scatter plot (grafik titik)

**Penjelasan per bagian:**
- `plt.scatter()` = fungsi untuk buat scatter plot
- `y_test` = sumbu X (nilai asli)
- `y_pred` = sumbu Y (prediksi)
- `alpha=0.5` = transparansi 50% (agar tidak terlalu pekat)

**Hasil:** Grafik titik-titik yang menunjukkan hubungan prediksi vs nilai asli

**Kenapa perlu:** Visualisasi membantu lihat pola prediksi

---

### `plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)`
**Apa itu:** Buat garis diagonal (garis prediksi sempurna)

**Penjelasan per bagian:**
- `plt.plot()` = fungsi untuk buat garis
- `[y_test.min(), y_test.max()]` = sumbu X: dari nilai terkecil ke terbesar
- `[y_test.min(), y_test.max()]` = sumbu Y: dari nilai terkecil ke terbesar
- `'r--'` = warna merah (r), garis putus-putus (--)
- `lw=2` = line width (ketebalan garis) = 2

**Hasil:** Garis diagonal merah putus-putus

**Kenapa perlu:** Garis ini menunjukkan "prediksi sempurna" (prediksi = nilai asli). Titik yang dekat dengan garis = prediksi bagus

---

### `plt.xlabel('Harga Asli')`
**Apa itu:** Beri label untuk sumbu X

**Fungsi:** Label membantu paham apa yang ditampilkan

---

### `plt.ylabel('Prediksi')`
**Apa itu:** Beri label untuk sumbu Y

**Fungsi:** Label membantu paham apa yang ditampilkan

---

### `plt.title('Prediksi vs Nilai Asli')`
**Apa itu:** Beri judul untuk grafik

**Fungsi:** Judul menjelaskan apa yang ditampilkan

---

### `plt.legend()`
**Apa itu:** Tampilkan legend (keterangan)

**Fungsi:** Menampilkan keterangan untuk setiap elemen grafik

---

### `plt.grid(True, alpha=0.3)`
**Apa itu:** Tampilkan grid (garis bantu) dengan transparansi

**Penjelasan per bagian:**
- `plt.grid()` = fungsi untuk tampilkan grid
- `True` = aktifkan grid
- `alpha=0.3` = transparansi 30% (agar tidak terlalu mencolok)

**Fungsi:** Grid membantu membaca nilai di grafik

---

### `plt.hist(error, bins=30)`
**Apa itu:** Buat histogram (grafik distribusi)

**Penjelasan per bagian:**
- `plt.hist()` = fungsi untuk buat histogram
- `error` = data yang mau dihistogram (selisih prediksi vs asli)
- `bins=30` = bagi menjadi 30 kelompok

**Hasil:** Grafik batang yang menunjukkan distribusi error

**Kenapa perlu:** Lihat sebaran error - apakah terpusat di 0 (bagus) atau tersebar (kurang bagus)

---

### `plt.tight_layout()`
**Apa itu:** Atur layout agar tidak overlap

**Fungsi:** Memastikan semua elemen grafik terlihat dengan baik

---

### `plt.show()`
**Apa itu:** Tampilkan grafik

**Fungsi:** Menampilkan semua grafik yang sudah dibuat

**Kenapa perlu:** Tanpa ini, grafik tidak akan muncul

---

## 🎯 PREDIKSI DATA BARU

### `rumah_baru = np.array([[...]])`
**Apa itu:** Buat array NumPy dengan data rumah baru

**Penjelasan per bagian:**
- `np.array()` = fungsi untuk buat array NumPy
- `[[...]]` = nested list (list dalam list)
  - List luar = array
  - List dalam = satu data (satu rumah)
- Setiap angka = nilai fitur rumah

**Struktur:**
```python
[[fitur1, fitur2, fitur3, ...]]  # satu rumah, banyak fitur
```

**Kenapa perlu:** Butuh format array untuk input ke model

---

### `model.predict(rumah_baru)[0]`
**Apa itu:** Prediksi harga rumah baru, ambil hasil pertama

**Penjelasan per bagian:**
- `model.predict(rumah_baru)` = prediksi (hasilnya array)
- `[0]` = ambil elemen pertama (karena hanya satu rumah)

**Hasil:** Satu angka (prediksi harga)

**Kenapa perlu:** Ini aplikasi nyata model - prediksi data baru

---

## 💡 RINGKASAN

**Import:**
- `import ... as ...` = import library dengan alias
- `from ... import ...` = import bagian tertentu

**Data:**
- `pd.DataFrame()` = buat tabel
- `df['kolom']` = akses kolom
- `df.drop()` = hapus kolom/baris
- `.shape` = dimensi data

**Model:**
- `LinearRegression()` = buat model
- `.fit()` = train model
- `.predict()` = prediksi
- `.coef_` = koefisien
- `.intercept_` = intercept

**Evaluasi:**
- `mean_squared_error()` = hitung MSE
- `r2_score()` = hitung R²

**Visualisasi:**
- `plt.figure()` = buat canvas
- `plt.subplot()` = buat subplot
- `plt.scatter()` = grafik titik
- `plt.plot()` = grafik garis
- `plt.hist()` = histogram
- `plt.show()` = tampilkan

---

**Selamat belajar! 🚀**

