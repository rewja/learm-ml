1. Apa itu ML
Cabang dari AI yang fokus pada bagaimana komputer dapat belajar dari data dan menemukan pola tanpa harus diprogram aturan eksplisit satu per satu.

alur kerja = input - model - output 

pembatu model = 
dataset :
- training
- validation
- testing
loss function 
optimizer

2. Perbedaan AI vs ML vs DL
AI (Artificial Intelligence) = mesin peniru kecerdasan manusia :
- logika
- aturan
- pencarian
ML (Machine Learning) = bagian dari AI, subset dari AI 
- pembelajaran dari data
DL (Deep Learning) = subset dari ML 
- neural network dengan banyak lapisan (deep neural networks)
untuk data kompleks (gambar, suara, dan text)


3. Mengapa ML penting
- digunakan di berbagai bidang :
rekomendasi film, deteksi penyakit, analisis keuangan, pengenalan wajah
- menangani data dalam jumlah besar
- menjadi dasar pengembangan teknologi modern
- mengotomatisasi proses pengambilan keputusan → skala besar & real-time, misalnya 10 juta rekomendasi per detik.

RANGKUMAN:
- ML bekerja dengan pola dari data 
- Semakin banyak kualitas datanya, semakin baik predksi model
ML = subset AI
DL = subset ML
- Neural Network dalam DL biasanya memanfaatkan GPU/TPU karena perhitungannya besar


KATEGORI UTAMA ALGORITMA ML
1. Supervised Learning (Dengan Pengawasan)
Definisi = Mengunakan label untuk melatih ML
contoh algoritma:
~ Linear Regression -> memprediksi nilai kontinu (harga rumah)
~ Logistic Regression -> klasifikasi biner (spam vs non-spam)
~ Decision Tree -> membuat pohon keputusan dari data (kredit bank - apakah nasabah layak diberi pinjaman)
~ Random Forest -> kumpulan banyak pohon keputusan untuk hasil lebih stabil (sistem pendeteksi penipuan)
~ Support Vector Machine (SVM) -> mencari garis pemisah terbaik kelas (pengenalan wajah di sistem keamanan)
~ K-Nearest Neighbors -> klasifikasi berdasarkan tetangga terdekat (rekomendasi produk berdasarkan kesamaan preferensi pengguna)

Alur = Data training -> model -> prediksi -> bandingkan dengan label -> diperbaiki lewat loss function + optimizer

# Setelah diperbaiki via optimizer → ulang terus sampai loss cukup kecil.
Ini disebut training loop.

2. Unsupervised Leaning (Tanpa Pengawasan)
Defini = Tidak menggunakan label, fokus pada pola
contoh algpritma:
- K-Means Clustering -> mengelompokan data ke dalam cluster (sistem telekomunikasi, e-commerce, pola nasabah bank)
- Hierchical Clustering -> membuat hierarki kelompok data (sistem bioinformatika/genetika)
- Principal Component Analysis (PCA) -> reduksi dimensi untuk menyederhanakan data (face recognition, system saham untuk menganalisis risiko protofolio, medical imaging system)

Alur = Model mencari pola, struktur atau kelompok dalam data 

3. Reinforcement Learning (Penguatan)   
Definisi = Belajar dari trial and error dengan umpan balik berupa reward atau penalty
contoh algoritma:
- Q-Learning -> Agen belajar memilih aksi terbaik berbadarkan nilai Q (robot asisten)
- Deep Q-Network (DQN) -> Gabungan Q-Lerning dengan neural network (lawan bot)
- Policy Gradient Methods -> langsung mengoptimalkan kebijakan aksi (sistem navigasi)

Alur =  Agent(model) -> aksi -> lingkungan -> feedback (reward/penalty) -> perbaikan strategi

ANALOGI SEDERHANA 
- Supervised Learning = Belajar denan guru
arahan yang jelas (guru + jawaban)
- Unsupervised Learning = Eksplorasi sendiri
- Reinforcement Learning = Belajar dari pengalaman dan konsekuensi

DATASET
Training = model melakukan pembelajaran, disesuaikan engan apa kategori algoritmanya:
- supervised dengan data berlabel (terdapat jawaban yang benar)
- unsuvervised tanpa label (tidak ada jawaban benar, belajar melalui pola)
# reinconforcement datasetnya beda konsep, model berupa agent yang belajar dari interaksi lingkunagan (pengalaman dengan reward atau penalty)

Validation Dataset = model melakukan pengetesan 
- mengukur peforma
- mencegah overfitting: terpaku pada data training, gagal pada data baru
- membantu memilih pamameter terbaik 

Testing = Mengukur peforma akhir model

KERANGKA DASAR SISTEM ML
1. Input -> data mentah
2. Model -> algoritma belajar mengelola data mentah tersebut
3. Output -> pridiksi/hasil

data masuk -> diproses -> hasil keluar

LOSS FUNCTION + OPTIMIZER 
1. Loss Function (alat ukur kesalahan)
alat ukur prediksi model apakah prediksinya bagus atau jelek
contoh umum:
- MSE
- Cross-Entropy
2. Optimizer (alat perbaikan kesalahan)
memperbaiki nilai prediksi untuk model
contoh umum:
SGD
Adam
RMSProp