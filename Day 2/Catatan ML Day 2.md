StandardScaler = alat untuk menyamakan skala semua fitur

tujuan:
- semua fitur adil
- tidak ada yang "terlalu dominan"

StandardScaler ngubah tiap fitur jadi
mean = 0
std = 1

std (standard deviation) = ukuran seberapa nyebar data dari mean

std kecil → data rapat
std besar → data nyebar

StandardScaler:
- kurangi mean
- bagi std
→ semua fitur punya skala setara

rumus:
x_scaled = (x - mean) / std

kapan pakai StandardScaler?
- Linear Regression
- Ridge / Lasso
- KNN
- SVM

❌ tidak perlu scaling untuk tree-based model
(Decision Tree, Random Forest)

R² = 1 sempurna
MSE = 0 sempurna

R² tinggi ≠ model selalu bagus
MSE kecil tergantung skala target

alpha = untuk kasih aturan / rem pada model

alpha besar → underfitting
alpha kecil → overfitting

gridSearch = mesin pencari hyperparameter
dipakai di data train (pakai cross-validation)

ML itu BUKAN milih satu model.
ML itu proses nyoba, ngebandingin, dan milih model yang paling cocok.

Tidak ada model terbaik untuk semua masalah (No Free Lunch).