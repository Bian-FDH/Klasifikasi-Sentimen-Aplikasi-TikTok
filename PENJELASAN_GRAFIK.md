# 📊 PENJELASAN GRAFIK ANALISIS SENTIMEN TIKTOK

## Ringkasan Dataset 200 Sampel

```
Total Sampel: 200
├── Sentimen Positif: 100 (50%)
└── Sentimen Negatif: 100 (50%)
```

---

## 📈 GRAFIK 1: DISTRIBUSI TOTAL SENTIMEN (Bar Chart)

### Deskripsi:
Bar chart ini menampilkan jumlah total review untuk setiap kategori sentimen.

### Data:
- **Sentimen Positif**: 100 sampel (ditampilkan dengan warna hijau #00D084)
- **Sentimen Negatif**: 100 sampel (ditampilkan dengan warna merah #FF3B30)

### Interpretasi:
- Dataset memiliki **distribusi yang sangat seimbang** (perfectly balanced)
- Tidak ada bias terhadap satu kelas tertentu
- Ideal untuk training model machine learning karena kedua kelas memiliki representasi yang sama
- Persentase: Masing-masing sentimen 50%

### Keuntungan:
✅ Model tidak akan bias ke satu kelas
✅ Metrik evaluasi seperti accuracy lebih reliable
✅ Balanced dataset lebih mudah diinterpretasikan

---

## 🥧 GRAFIK 2: PROPORSI SENTIMEN (Pie Chart)

### Deskripsi:
Pie chart menunjukkan proporsi persentase setiap sentimen dari total 200 sampel.

### Data:
- **Positif**: 50.0% (100 dari 200)
- **Negatif**: 50.0% (100 dari 200)

### Interpretasi:
- Pembagian yang sempurna 50-50
- Setiap jenis sentimen memiliki bobot yang sama dalam dataset
- Cocok untuk pembelajaran supervised karena class balance sempurna

### Implikasi Bisnis:
- Review aplikasi TikTok memiliki feedback yang seimbang antara positif dan negatif
- Aplikasi memiliki kelebihan dan kekurangan yang sama-sama dirasakan pengguna

---

## 📊 GRAFIK 3: DISTRIBUSI PANJANG REVIEW (Histogram)

### Deskripsi:
Histogram menunjukkan distribusi frekuensi berdasarkan panjang review (jumlah karakter).

### Data Statistik Panjang Review:
```
SEMUA REVIEW:
  Minimum: [nilai] karakter
  Maksimum: [nilai] karakter
  Rata-rata: [nilai] karakter
  Median: [nilai] karakter

REVIEW POSITIF:
  Rata-rata: [nilai] karakter
  Median: [nilai] karakter

REVIEW NEGATIF:
  Rata-rata: [nilai] karakter
  Median: [nilai] karakter
```

### Interpretasi:
- **Bentuk Distribusi**: Menunjukkan pola distribusi panjang ulasan
- **Peak**: Frekuensi tertinggi menunjukkan panjang ulasan yang paling umum
- **Spread**: Menunjukkan variabilitas panjang review dalam dataset

### Insights:
- Jika histogram condong ke kanan (right-skewed): Ada review yang sangat panjang
- Jika histogram condong ke kiri (left-skewed): Ada banyak review singkat
- Jika simetris: Panjang review tersebar merata

### Perbandingan Sentimen:
- Review positif vs negatif mungkin memiliki panjang rata-rata yang berbeda
- Biasanya review negatif lebih panjang (pengguna lebih detail saat complain)

---

## 📋 GRAFIK 4: STATISTIK DESKRIPTIF (Summary Box)

### Deskripsi:
Box ini menampilkan ringkasan statistik lengkap dari dataset.

### Informasi yang Ditampilkan:

#### A. Total Sampel dan Distribusi:
- Total: 200 sampel
- Positif: 100 (50%)
- Negatif: 100 (50%)

#### B. Statistik Panjang Review (Semua):
- **Min**: Panjang review terpendek
- **Max**: Panjang review terpanjang
- **Rata-rata**: Mean dari semua review
- **Median**: Nilai tengah dari semua review

#### C. Statistik Panjang Review Positif:
- **Rata-rata**: Mean review positif
- **Median**: Nilai tengah review positif

#### D. Statistik Panjang Review Negatif:
- **Rata-rata**: Mean review negatif
- **Median**: Nilai tengah review negatif

### Interpretasi:
- Membandingkan panjang review antar sentimen
- Jika review negatif lebih panjang: Users lebih detail mengeluh
- Jika review positif lebih panjang: Users lebih terperinci saat memuji

---

## 🎯 GRAFIK 5: CONFUSION MATRIX (Model Performance)

### Deskripsi:
Confusion matrix menunjukkan performa model klasifikasi pada test set (20% dari data = 40 sampel).

### Struktur Matrix:
```
                 PREDICTED
              Negatif  Positif
ACTUAL  Negatif  TN      FP
        Positif  FN      TP
```

### Komponen:
- **TP (True Positive)**: Model benar prediksi Positif (benar-benar positif)
- **TN (True Negative)**: Model benar prediksi Negatif (benar-benar negatif)
- **FP (False Positive)**: Model salah prediksi Positif (harusnya Negatif)
- **FN (False Negative)**: Model salah prediksi Negatif (harusnya Positif)

### Metrik Perhitungan:
```
Accuracy = (TP + TN) / Total
Precision (Positif) = TP / (TP + FP)
Recall (Positif) = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
```

### Interpretasi Warna:
- **Biru tua**: Nilai tinggi (prediksi benar banyak)
- **Biru muda**: Nilai rendah (prediksi salah sedikit)

### Expected Performance (untuk 40 test samples):
- Akurasi biasanya: 80-85% untuk dataset kecil
- Berarti: ~32-34 prediksi benar dari 40 sampel

### Contoh Interpretasi:
Jika matrix menunjukkan:
```
    Negatif  Positif
Negatif  18    2       (18 benar negatif, 2 false positif)
Positif   1   19       (1 false negatif, 19 benar positif)
```
Akurasi = (18+19)/40 = 92.5%

---

## 🔤 GRAFIK 6: TOP 10 FITUR SENTIMEN (Feature Importance)

### Deskripsi:
Bar horizontal menunjukkan 10 kata-kata (features) paling penting untuk memprediksi setiap sentimen.

### TOP FITUR SENTIMEN POSITIF (Hijau):
Kata-kata yang sangat mengindikasikan review positif:
- Contoh umum: "excellent", "great", "love", "amazing", "wonderful"
- "easy", "smooth", "good", "helpful", "best"

### TOP FITUR SENTIMEN NEGATIF (Merah):
Kata-kata yang sangat mengindikasikan review negatif:
- Contoh umum: "bug", "crash", "problem", "bad", "slow"
- "issue", "fail", "error", "difficult", "poor"

### Cara Membaca Grafik:
- **X-axis (Log Probability)**: Seberapa penting fitur tersebut
- **Nilai lebih tinggi**: Kata-kata yang lebih signifikan untuk kelas tersebut
- **Urutan top-down**: Ranking dari paling penting ke paling kurang penting

### Interpretasi:
- Model menggunakan kata-kata ini sebagai indikator utama untuk prediksi
- Semakin tinggi bar, semakin kuat pengaruh kata tersebut
- Ini adalah "signature words" untuk setiap sentimen

### Kegunaan:
1. **Understanding Model**: Mengerti bagaimana model membuat keputusan
2. **Validation**: Mengecek apakah fitur penting masuk akal
3. **Domain Insight**: Memahami drivers utama sentimen pengguna
4. **Feature Engineering**: Ide untuk menambah fitur baru yang lebih baik

---

## 📊 GRAFIK 7: REVIEW LENGTH COMPARISON (Box Plot & Violin Plot)

### Deskripsi Box Plot:
Menampilkan distribusi panjang review per sentimen dengan statistik quartile.

**Komponen Box Plot:**
- **Garis bawah (Q1)**: 25% data di bawah nilai ini
- **Garis tengah (Median)**: 50% data di bawah/atas nilai ini
- **Garis atas (Q3)**: 75% data di bawah nilai ini
- **Dots (Outliers)**: Nilai yang jauh dari rentang normal

### Deskripsi Violin Plot:
Menunjukkan distribusi probabilitas panjang review (smooth version of histogram).

### Interpretasi:
- **Violin lebih lebar**: Ada banyak review dengan panjang tersebut
- **Violin lebih sempit**: Sedikit review dengan panjang tersebut
- **Membandingkan bentuk**: Melihat pola berbeda antara positif dan negatif

### Insights:
- Jika review positif box lebih sempit: Review positif lebih konsisten panjangnya
- Jika review negatif lebih tinggi: Review negatif cenderung lebih panjang
- Outliers menunjukkan review yang sangat panjang atau sangat singkat

---

## 🎓 GRAFIK 8: MODEL PERFORMANCE METRICS

### Deskripsi:
3 bar chart menampilkan metrics evaluasi model (Precision, Recall, F1-Score).

### Precision (Grafik Kiri):
**Pertanyaan**: "Dari semua prediksi Positif, berapa yang benar-benar Positif?"
- Formula: TP / (TP + FP)
- Range: 0-1 (atau 0-100%)
- Interpretasi:
  - 0.9 = 90% prediksi positif benar-benar positif
  - Penting ketika False Positive mahal

### Recall (Grafik Tengah):
**Pertanyaan**: "Dari semua yang benar-benar Positif, berapa yang berhasil dideteksi?"
- Formula: TP / (TP + FN)
- Range: 0-1 (atau 0-100%)
- Interpretasi:
  - 0.85 = 85% dari positif yang sebenarnya berhasil dideteksi
  - Penting ketika False Negative mahal

### F1-Score (Grafik Kanan):
**Pertanyaan**: "Balance antara Precision dan Recall?"
- Formula: 2 × (Precision × Recall) / (Precision + Recall)
- Range: 0-1 (atau 0-100%)
- Interpretasi:
  - Best untuk balanced evaluation
  - Menghindari model yang hanya bagus di satu metrik

### Benchmarks:
- **0.6-0.7**: Cukup baik
- **0.7-0.8**: Baik
- **0.8-0.9**: Sangat baik
- **>0.9**: Excellent

---

## 📌 RINGKASAN INSIGHTS KESELURUHAN

### Dataset Characteristics:
1. ✅ **Balanced**: 50% positif, 50% negatif
2. ✅ **Diverse**: Review dengan panjang yang bervariasi
3. ✅ **Representative**: Mencakup berbagai aspek aplikasi TikTok

### Model Performance:
1. **Training Data**: 160 sampel (80%)
2. **Testing Data**: 40 sampel (20%)
3. **Algorithm**: Multinomial Naive Bayes
4. **Features**: TF-IDF dengan max 100 features
5. **Expected Accuracy**: 80-85%

### Key Findings:
1. Review positif dan negatif memiliki pola kata yang berbeda jelas
2. Top features mengidentifikasi signature words untuk setiap sentimen
3. Panjang review mungkin berbeda antara positif dan negatif
4. Model cukup reliable untuk klasifikasi otomatis

### Business Recommendations:
1. 📈 Monitor sentiment trend dari reviews baru
2. 🔍 Fokus pada fitur-fitur yang drive negative sentiment
3. 📋 Leverage positive features untuk marketing
4. 🎯 Gunakan insights untuk product improvement

---

## 🔧 TECHNICAL NOTES

### Preprocessing:
- Text dikonversi lowercase
- Stopwords dihapus
- TF-IDF normalization applied

### Model Details:
- Algorithm: Multinomial Naive Bayes
- Kernel: Additive smoothing (Laplace: 1.0)
- Independence Assumption: Yes

### Limitations:
⚠️ Negation handling tidak sempurna (misalnya "not bad" bisa terklasifikasi salah)
⚠️ Sarcasm tidak terdeteksi
⚠️ Dataset kecil mungkin tidak generalize dengan sempurna

---

## 📚 HOW TO USE THESE INSIGHTS

1. **Untuk Eksekutif**: Lihat Grafik 1 & 2 untuk gambaran umum
2. **Untuk Data Scientist**: Perhatikan Grafik 5-8 untuk model insights
3. **Untuk Product Manager**: Fokus pada Grafik 6 untuk improvement areas
4. **Untuk Marketing**: Gunakan Grafik 6 positif untuk messaging

---

**Generated**: 26 Juli 2026
**Version**: 1.0
