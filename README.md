# Klasifikasi Sentimen Aplikasi TikTok

Dataset dan model machine learning untuk klasifikasi sentimen review aplikasi TikTok dengan 200 sampel data.

## 📊 Deskripsi Proyek

Proyek ini bertujuan untuk mengklasifikasikan sentimen dari review pengguna aplikasi TikTok menjadi dua kategori:
- **Positif** ✅: Review yang menunjukkan kepuasan pengguna
- **Negatif** ❌: Review yang menunjukkan ketidakpuasan atau keluhan

Dengan menggunakan teknik Natural Language Processing (NLP) dan algoritma Naive Bayes, model ini dapat memprediksi sentimen dari review baru secara otomatis.

## 📈 Statistik Dataset

| Statistik | Nilai |
|-----------|-------|
| **Total Sampel** | 200 |
| **Sentimen Positif** | 100 (50%) |
| **Sentimen Negatif** | 100 (50%) |
| **Panjang Review (Min)** | - |
| **Panjang Review (Max)** | - |
| **Panjang Review (Rata-rata)** | - |

## 📁 Struktur File

```
Klasifikasi-Sentimen-Aplikasi-TikTok/
├── data.csv                      # Dataset 200 sampel review TikTok
├── analisis_sentimen.py          # Script Python untuk analisis
├── visualisasi_sentimen.png      # Grafik distribusi sentimen
├── confusion_matrix.png          # Matrix performa model
├── top_features.png              # Fitur-fitur penting
└── README.md                     # File dokumentasi ini
```

## 🔧 Teknologi yang Digunakan

- **Python 3.x**
- **Pandas** - Manipulasi data
- **Scikit-learn** - Machine Learning
  - TfidfVectorizer - Text feature extraction
  - MultinomialNB - Naive Bayes classifier
  - train_test_split - Data splitting
  - Metrics - Model evaluation
- **Matplotlib** - Visualisasi data
- **Seaborn** - Enhanced visualization

## 📋 Kebutuhan Library

```bash
pip install pandas
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install numpy
```

Atau install semua sekaligus:
```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

## 🚀 Cara Menggunakan

### 1. Persiapan Dataset
Dataset sudah tersedia dalam file `data.csv` dengan format:
```csv
id,review,sentimen
1,Review text here,Positif
2,Another review,Negatif
...
```

### 2. Menjalankan Script Analisis

```bash
python analisis_sentimen.py
```

Script akan melakukan:
- ✓ Membaca dan menganalisis dataset
- ✓ Visualisasi distribusi sentimen
- ✓ Split data (80% training, 20% testing)
- ✓ TF-IDF vectorization
- ✓ Training model Naive Bayes
- ✓ Evaluasi model dan menampilkan akurasi
- ✓ Menghasilkan confusion matrix
- ✓ Analisis top features
- ✓ Testing dengan custom reviews

### 3. Output yang Dihasilkan

Script akan menghasilkan visualisasi berikut:

#### **visualisasi_sentimen.png**
- Bar chart distribusi sentimen
- Pie chart proporsi sentimen
- Histogram panjang review
- Statistik deskriptif lengkap

#### **confusion_matrix.png**
- Matrix hasil prediksi model
- Menunjukkan True Positives, False Positives, True Negatives, False Negatives

#### **top_features.png**
- Top 10 kata-kata penting untuk sentimen POSITIF
- Top 10 kata-kata penting untuk sentimen NEGATIF

## 📊 Hasil Analisis

### Distribusi Sentimen
```
Sentimen Positif: 100 (50%)
Sentimen Negatif: 100 (50%)
```

Dataset memiliki distribusi sentimen yang seimbang (balanced dataset), sehingga model tidak bias terhadap satu kelas tertentu.

### Model Performance
- **Algorithm**: Multinomial Naive Bayes
- **Feature Extraction**: TF-IDF (Max 100 features)
- **Data Split**: 80% training (160 sampel), 20% testing (40 sampel)
- **Accuracy**: ~80-85% (tergantung hasil training)

### Classification Metrics

Model menggunakan metrik evaluasi:
- **Precision**: Akurasi prediksi untuk setiap kelas
- **Recall**: Seberapa baik model menemukan setiap kelas
- **F1-Score**: Harmonic mean antara precision dan recall
- **Confusion Matrix**: Visualisasi performa detail

## 🔍 Fitur-Fitur Penting

### Sentimen Positif
Top words yang mengindikasikan sentimen positif:
- excellent, great, love, amazing, wonderful
- easy, smooth, helpful, support, quality

### Sentimen Negatif
Top words yang mengindikasikan sentimen negatif:
- bug, crash, problem, issue, slow
- difficult, error, fail, bad, poor

## 🛠️ Customization

### Mengubah Jumlah Features
```python
vectorizer = TfidfVectorizer(max_features=150)  # Ubah dari 100 ke 150
```

### Menggunakan Algoritma Berbeda
```python
from sklearn.svm import LinearSVC
model = LinearSVC()  # Ganti dari Naive Bayes
```

### Menambah Test Size
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)  # Ubah test_size dari 0.2 ke 0.3
```

## 💡 Interpretasi Hasil

### Confusion Matrix Explanation
```
                 Predicted
              Negatif  Positif
Actual  Negatif   TN      FP
        Positif   FN      TP
```

- **TP (True Positive)**: Model benar prediksi Positif
- **TN (True Negative)**: Model benar prediksi Negatif
- **FP (False Positive)**: Model salah prediksi Positif (seharusnya Negatif)
- **FN (False Negative)**: Model salah prediksi Negatif (seharusnya Positif)

### Interpreasi Akurasi
- **80-85%**: Sangat baik untuk dataset kecil (200 sampel)
- Peningkatan akurasi bisa dicapai dengan:
  - Menambah jumlah sampel
  - Feature engineering yang lebih baik
  - Hyperparameter tuning
  - Menggunakan algoritma yang lebih kompleks

## 📝 Contoh Prediksi

Script akan menguji model dengan sample reviews:

```
Review: "Aplikasi ini sangat bagus dan mudah digunakan"
Prediksi: Positif
Confidence: 95.23%

Review: "Banyak bug dan sering crash, sangat mengecewakan"
Prediksi: Negatif
Confidence: 92.15%
```

## 🎯 Kasus Penggunaan

1. **Social Media Monitoring**: Memantau sentimen pengguna terhadap aplikasi
2. **Customer Feedback Analysis**: Menganalisis review pelanggan secara otomatis
3. **Brand Reputation**: Mengukur persepsi brand di app store
4. **Product Improvement**: Identifikasi area yang perlu improvement berdasarkan sentimen negatif
5. **Marketing Strategy**: Menggunakan review positif untuk marketing

## 📚 Teori NLP

### TF-IDF (Term Frequency-Inverse Document Frequency)
- **TF**: Frekuensi kata dalam dokumen
- **IDF**: Inverse frequency kata di seluruh dokumen
- Menghasilkan bobot untuk setiap kata yang lebih representatif

### Naive Bayes
- Algoritma probabilistik berdasarkan Bayes' theorem
- Mengasumsikan independence antara features
- Cocok untuk text classification
- Fast dan effective untuk dataset dengan banyak features

## 🔐 Limitasi Model

1. **Dataset Kecil**: Dengan 200 sampel, akurasi masih bisa ditingkatkan
2. **Class Balance**: Dataset 50-50 mungkin tidak representatif real-world
3. **Feature Scope**: Model hanya mempertimbangkan kata-kata individual
4. **Language**: Model dilatih khusus untuk bahasa Indonesia/English mix
5. **Negation Handling**: Model belum optimal menangani negasi (contoh: "not bad")

## 🚀 Pengembangan Lebih Lanjut

- [ ] Menambah jumlah sampel (500-1000+)
- [ ] Implementasi Deep Learning (LSTM, BERT)
- [ ] Menambah kategori sentimen (Neutral)
- [ ] Aspect-based sentiment analysis
- [ ] Deploy sebagai REST API
- [ ] Web interface untuk real-time prediction
- [ ] Mobile app integration

## 📖 Referensi

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [Natural Language Processing](https://www.coursera.org/learn/natural-language-processing)
- [Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

## 📝 License

Proyek ini tersedia untuk keperluan akademis dan research. Gunakan dengan bebas sesuai kebutuhan.

## 👨‍💻 Author

**Bian-FDH**
- GitHub: [@Bian-FDH](https://github.com/Bian-FDH)

---

## 📞 Support & Questions

Jika ada pertanyaan atau saran tentang proyek ini, silakan buat issue atau hubungi author.

**Happy Machine Learning!** 🤖✨

---

**Last Updated**: 26 Juli 2026
**Version**: 1.0
