# Analisis Sentimen Ulasan Pengguna Aplikasi Grab

Analisis sentimen terhadap ulasan pengguna aplikasi **Grab** di Google Play Store, menggunakan pendekatan *lexicon-based labeling* dan perbandingan beberapa model *machine learning* & *deep learning* untuk klasifikasi sentimen (positif, negatif, netral).


## Tentang Project

Project ini bertujuan untuk memahami persepsi pengguna terhadap aplikasi Grab (fokus pada layanan *ride-hailing*, `com.grabtaxi.passenger`) berdasarkan ulasan yang mereka tulis di Google Play Store. Alur kerja project mencakup:

1. **Scraping data** ulasan langsung dari Google Play Store.
2. **Preprocessing teks** (cleaning, normalisasi kata gaul/slang, tokenisasi, stopword removal, stemming).
3. **Pelabelan sentimen otomatis** menggunakan pendekatan *lexicon-based* (kamus kata positif & negatif Bahasa Indonesia).
4. **Visualisasi data** (word cloud, distribusi kelas, pie chart sentimen).
5. **Pelatihan & perbandingan beberapa model klasifikasi**, mulai dari *machine learning* klasik hingga *deep learning*.
6. **Inferensi** — menguji model pada kalimat baru untuk memprediksi sentimennya.

## Struktur Project

```
Analisis-Sentimen-Grab/
├── scrapping_data.ipynb      # Notebook untuk scraping ulasan dari Google Play Store
├── training_model.ipynb     # Notebook utama: preprocessing, pelabelan, training & evaluasi model
├── ulasan_grab.csv          # Dataset hasil scraping (~50.000 ulasan)
├── slangwords.json          # Kamus kata gaul/slang Bahasa Indonesia untuk normalisasi teks
├── requirements.txt         # Daftar dependency Python yang dibutuhkan
└── README.md
```

## Dataset

Dataset dikumpulkan langsung melalui **web scraping** menggunakan library `google-play-scraper`, mengambil hingga **50.000 ulasan terbaru** aplikasi Grab (`com.grabtaxi.passenger`) berbahasa Indonesia dari Google Play Store.

## Alur Pengerjaan

### 1. Scraping Data (`ScrappingData.ipynb`)
Mengambil ulasan pengguna dari Google Play Store menggunakan `google_play_scraper`, lalu menyimpannya ke `ulasan_grab.csv`.

### 2. Preprocessing (`Training_Model.ipynb`)
- **Cleaning** : menghapus mention, hashtag, tautan/URL, angka, dan tanda baca.
- **Case folding** : menyeragamkan huruf menjadi huruf kecil.
- **Normalisasi slang** : mengganti kata gaul/tidak baku menggunakan kamus `slangwords.json`.
- **Tokenizing & stopword removal** : memecah kalimat menjadi kata dan membuang kata umum yang tidak informatif.
- **Stemming** : menggunakan library **Sastrawi** untuk mengubah kata ke bentuk dasarnya.

### 3. Pelabelan Sentimen
Menggunakan pendekatan **lexicon-based** dengan kamus kata positif & negatif Bahasa Indonesia, setiap ulasan diberi skor polaritas untuk menentukan label **positif**, **negatif**, atau **netral**.

### 4. Visualisasi Data
- Pie chart distribusi sentimen
- Word cloud untuk masing-masing kelas sentimen (positif, negatif, netral)
- Grafik distribusi kelas

### 5. Ekstraksi Fitur & Pelatihan Model
Data diekstraksi fiturnya menggunakan **TF-IDF** dan **Bag of Words (BoW)**, lalu diseimbangkan dengan teknik oversampling **SMOTE**. Beberapa model dilatih dan dibandingkan performanya:

| Model | Ekstraksi Fitur | Split Data |
|---|---|---|
| Random Forest | TF-IDF | 70/30 |
| Random Forest | BoW | 80/20 |
| Logistic Regression | BoW | 75/25 |
| Decision Tree | TF-IDF | 70/30 |
| **GRU (Deep Learning)** | Tokenization + Padding | 80/20 |

### 6. Evaluasi & Inferensi
Model dievaluasi menggunakan **accuracy**, **precision**, **recall**, dan **F1-score**, lalu diuji langsung pada kalimat baru untuk memprediksi sentimennya secara *real-time*.

## Tech Stack

- **Python**
- **Pandas** & **NumPy** : manipulasi data
- **NLTK** & **Sastrawi** : pemrosesan bahasa alami (tokenizing, stopword, stemming Bahasa Indonesia)
- **Scikit-learn** : TF-IDF, BoW, model klasik (Random Forest, Logistic Regression, Decision Tree), SMOTE
- **TensorFlow / Keras** : model Deep Learning (GRU)
- **Matplotlib**, **Seaborn**, **WordCloud** : visualisasi data
- **google-play-scraper** : scraping ulasan dari Google Play Store

## Cara Menjalankan

1. **Clone repository**
   ```sh
   git clone https://github.com/alfdmsr/Analisis-Sentimen-Grab.git
   cd Analisis-Sentimen-Grab
   ```

2. **Install dependencies**
   ```sh
   pip install -r requirements.txt
   pip install google-play-scraper Sastrawi imblearn
   ```

3. **(Opsional) Scraping ulang data terbaru**

   Jalankan `scrapping_data.ipynb` jika ingin mengambil data ulasan Grab terbaru. Jika ingin langsung memakai data yang sudah tersedia, lewati langkah ini dan gunakan `ulasan_grab.csv` yang sudah ada di repository.

4. **Jalankan notebook utama**
   ```sh
   jupyter notebook training_model.ipynb
   ```
   Jalankan seluruh cell secara berurutan dari atas ke bawah untuk melakukan preprocessing, pelabelan, pelatihan model, hingga inferensi.
