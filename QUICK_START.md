# 🚀 QUICK START - Klasifikasi Sentimen Aplikasi TikTok

Panduan cepat untuk memulai menggunakan proyek klasifikasi sentimen TikTok.

## 📋 Persyaratan

- Python 3.7+
- pip (Python package manager)

## ⚡ Instalasi (2 menit)

### 1. Clone Repository
```bash
git clone https://github.com/Bian-FDH/Klasifikasi-Sentimen-Aplikasi-TikTok.git
cd Klasifikasi-Sentimen-Aplikasi-TikTok
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Atau install manual:
```bash
pip install pandas scikit-learn matplotlib seaborn numpy
```

## 🎯 Penggunaan (3 cara)

### CARA 1: Analisis Lengkap (Recommended untuk pemula)
Jalankan script yang sudah jadi:

```bash
python analisis_sentimen.py
```

**Output yang dihasilkan:**
- `visualisasi_sentimen.png` - Overview 4 grafik
- `confusion_matrix.png` - Confusion matrix model
- `top_features.png` - Kata-kata penting
- Console output dengan statistik dan hasil

⏱️ **Waktu:** ~5-10 detik

---

### CARA 2: Contoh Lengkap (Untuk belajar)
Jalankan script dengan 13 contoh berbeda:

```bash
python example_usage.py
```

**Apa yang akan Anda pelajari:**
- Load dan explore dataset
- Analisis statistik
- Membuat visualisasi
- Persiapan data ML
- Feature extraction
- Training model
- Evaluasi performa
- Prediksi custom
- Error analysis
- Dan banyak lagi!

⏱️ **Waktu:** ~10-15 detik

---

### CARA 3: Kode Custom (Untuk advanced users)
Buat script Python sendiri:

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from visualizer import SentimenVisualizer

# Load data
df = pd.read_csv('data.csv')

# Persiapan
X = df['review']
y = df['sentimen']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Vectorization
vectorizer = TfidfVectorizer(max_features=100)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Training
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Prediksi
accuracy = model.score(X_test_tfidf, y_test)
print(f"Accuracy: {accuracy*100:.2f}%")

# Prediksi custom
review = "Aplikasi ini sangat bagus!"
pred = model.predict(vectorizer.transform([review]))[0]
print(f"Sentimen: {pred}")
```

---

## 📊 Memahami Grafik Hasil

### Grafik 1: Distribusi Sentimen
- **Menunjukkan:** Jumlah review positif dan negatif
- **Interpretasi:** 100 Positif + 100 Negatif = dataset seimbang ✓

### Grafik 2: Proporsi Sentimen
- **Menunjukkan:** Persentase setiap sentimen
- **Interpretasi:** 50% positif, 50% negatif

### Grafik 3: Panjang Review
- **Menunjukkan:** Distribusi panjang review
- **Interpretasi:** Review positif vs negatif punya panjang berbeda?

### Grafik 4: Statistik
- **Menunjukkan:** Ringkasan statistik lengkap
- **Interpretasi:** Min, max, rata-rata, median panjang review

### Grafik 5: Confusion Matrix
- **Menunjukkan:** Akurasi prediksi model
- **Interpretasi:** Berapa prediksi benar vs salah

### Grafik 6: Top Features
- **Menunjukkan:** Kata-kata penting untuk setiap sentimen
- **Interpretasi:** Apa yang membuat review positif/negatif?

---

## 🎓 Hasil yang Diharapkan

```
Dataset:
  Total Sampel: 200
  Positif: 100 (50%)
  Negatif: 100 (50%)

Model Performance:
  Accuracy: 80-85% (pada test set)
  Precision: ~0.80-0.85
  Recall: ~0.80-0.85
  F1-Score: ~0.80-0.85

Top Positive Words:
  excellent, great, amazing, good, best

Top Negative Words:
  bug, crash, problem, bad, slow
```

---

## 💡 Tips & Trik

### Tip 1: Prediksi Review Baru
```python
# Tanpa menjalankan seluruh script
new_review = "Aplikasi ini sangat menyenangkan"
pred = model.predict(vectorizer.transform([new_review]))[0]
confidence = model.predict_proba(vectorizer.transform([new_review]))[0].max()
print(f"Sentimen: {pred} (Confidence: {confidence*100:.2f}%)")
```

### Tip 2: Mengubah Parameter Model
```python
# Lebih banyak features
vectorizer = TfidfVectorizer(max_features=200)  # dari 100

# Mengubah test size
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)  # dari 0.2

# Algoritma berbeda
from sklearn.svm import LinearSVC
model = LinearSVC()
```

### Tip 3: Menyimpan Model
```python
import pickle

# Simpan model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
```

### Tip 4: Prediksi Multiple Reviews Sekaligus
```python
reviews = [
    "Sangat bagus!",
    "Banyak bug",
    "Algoritma recommended akurat"
]

predictions = model.predict(vectorizer.transform(reviews))
for review, pred in zip(reviews, predictions):
    print(f"{review} -> {pred}")
```

---

## 📁 Struktur File

```
Klasifikasi-Sentimen-Aplikasi-TikTok/
├── data.csv                      # Dataset (200 sampel)
├── analisis_sentimen.py          # Script utama
├── example_usage.py              # 13 contoh penggunaan
├── visualizer.py                 # Modul visualisasi
├── requirements.txt              # Dependencies
├── README.md                     # Dokumentasi lengkap
├── PENJELASAN_GRAFIK.md         # Penjelasan grafik detail
├── QUICK_START.md               # File ini
├── visualisasi_sentimen.png     # Output grafik
├── confusion_matrix.png         # Output grafik
├── top_features.png             # Output grafik
└── review_length_comparison.png # Output grafik
```

---

## 🤔 FAQ (Frequently Asked Questions)

### Q1: Dataset saya kecil, bagaimana cara meningkatkan akurasi?
**A:** 
- Tambah jumlah sampel (lebih banyak data = model lebih baik)
- Fine-tune parameter TF-IDF
- Gunakan algoritma yang lebih kompleks (SVM, Random Forest, Deep Learning)
- Lakukan feature engineering

### Q2: Bagaimana cara menggunakan untuk dataset lain?
**A:**
```python
# Ganti hanya 1 baris ini
df = pd.read_csv('dataset_anda.csv')  # Sesuaikan kolom: 'review', 'sentimen'

# Pastikan format sama:
# id, review, sentimen
# 1, "text here", Positif
```

### Q3: Akurasi rendah, apa yang salah?
**A:** Cek:
- ✓ Dataset cukup besar? (200+ sampel minimal)
- ✓ Label sudah benar? (Positif/Negatif)
- ✓ Review text sudah clean? (no missing values)
- ✓ Parameter sudah optimal? (max_features, test_size)

### Q4: Bagaimana cara deploy model ke production?
**A:** 
- Simpan model dengan pickle
- Buat API dengan Flask/FastAPI
- Deploy ke cloud (Heroku, AWS, GCP, dll)
- Lihat file advanced_deployment.py (jika tersedia)

### Q5: Apa bedanya Naive Bayes dengan algoritma lain?
**A:**
| Aspek | Naive Bayes | SVM | Random Forest | Deep Learning |
|-------|------------|-----|---------------|----------------|
| Speed | Cepat | Sedang | Sedang | Lambat |
| Akurasi | Baik | Sangat Baik | Sangat Baik | Excellent |
| Data needed | Sedikit | Sedang | Banyak | Banyak |
| Interpretasi | Mudah | Sulit | Sedang | Sangat Sulit |

---

## 🔧 Troubleshooting

### Error: ModuleNotFoundError: No module named 'pandas'
**Solusi:**
```bash
pip install pandas
```

### Error: No such file or directory: 'data.csv'
**Solusi:**
- Pastikan Anda di folder yang sama dengan data.csv
- Atau gunakan path lengkap: `pd.read_csv('/path/to/data.csv')`

### Error: Shape mismatch
**Solusi:**
- Pastikan training data sudah di-fit ke vectorizer
- Jangan lupa `vectorizer.fit_transform(X_train)` sebelum `transform(X_test)`

### Grafik tidak muncul
**Solusi:**
- Uncomment: `plt.show()` di akhir script
- Atau buka file PNG yang sudah tersimpan

---

## 📚 Resources Tambahan

### Belajar Lebih Lanjut:
- 📖 [Scikit-learn Documentation](https://scikit-learn.org/)
- 📖 [Pandas Documentation](https://pandas.pydata.org/)
- 📖 [Natural Language Processing](https://www.coursera.org/learn/natural-language-processing)
- 🎥 [YouTube: NLP dengan Python](https://www.youtube.com/results?search_query=NLP+python)

### Tools Berguna:
- 🔗 [Google Colab](https://colab.research.google.com/) - Jalan Python di browser
- 🔗 [Kaggle](https://www.kaggle.com/) - Dataset dan competition
- 🔗 [Hugging Face](https://huggingface.co/) - Pre-trained models

---

## ⏱️ Waktu Eksekusi Perkiraan

| Task | Waktu |
|------|-------|
| Load dataset | <1 detik |
| Exploratory analysis | 1-2 detik |
| Visualisasi | 2-3 detik |
| TF-IDF vectorization | 1-2 detik |
| Training model | <1 detik |
| Prediksi (40 sampel) | <1 detik |
| **TOTAL** | **~10-15 detik** |

---

## 🎯 Next Steps

1. ✅ **Jalankan** `python analisis_sentimen.py`
2. 📊 **Lihat** grafik hasil di folder ini
3. 📖 **Baca** PENJELASAN_GRAFIK.md untuk detail
4. 💻 **Eksperimen** dengan example_usage.py
5. 🚀 **Kustomisasi** untuk dataset Anda sendiri

---

## 🤝 Kontribusi

Punya ide untuk improvement?
- Fork repository ini
- Buat branch baru: `git checkout -b feature/amazing-feature`
- Commit changes: `git commit -m 'Add amazing feature'`
- Push ke branch: `git push origin feature/amazing-feature`
- Buat Pull Request

---

## 📞 Support

Ada pertanyaan atau issue?
- 🐛 [Buat Issue](https://github.com/Bian-FDH/Klasifikasi-Sentimen-Aplikasi-TikTok/issues)
- 💬 [Diskusi](https://github.com/Bian-FDH/Klasifikasi-Sentimen-Aplikasi-TikTok/discussions)
- 📧 Contact: [@Bian-FDH](https://github.com/Bian-FDH)

---

## 📝 License

Project ini tersedia untuk keperluan akademis dan research.

---

**Happy Machine Learning! 🚀✨**

*Last Updated: 26 Juli 2026*
*Version: 1.0*
