"""
CONTOH PENGGUNAAN - Klasifikasi Sentimen Aplikasi TikTok
File ini menunjukkan berbagai cara menggunakan dataset dan model
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from visualizer import SentimenVisualizer

print("=" * 70)
print("CONTOH PENGGUNAAN KLASIFIKASI SENTIMEN TIKTOK")
print("=" * 70)

# ============================================================================
# CONTOH 1: LOAD DAN EXPLORE DATASET
# ============================================================================
print("\n[CONTOH 1] Load dan Explore Dataset")
print("-" * 70)

# Load dataset
df = pd.read_csv('data.csv')

# Tampilkan 10 sampel pertama
print("\n10 Sampel Pertama Dataset:")
print(df.head(10).to_string())

# Informasi dataset
print(f"\n\nInfo Dataset:")
print(f"  Total Sampel: {len(df)}")
print(f"  Kolom: {list(df.columns)}")
print(f"  Tipe Data: {dict(df.dtypes)}")

# ============================================================================
# CONTOH 2: ANALISIS STATISTIK DATASET
# ============================================================================
print("\n\n[CONTOH 2] Analisis Statistik Dataset")
print("-" * 70)

# Distribusi sentimen
print("\nDistribusi Sentimen:")
print(df['sentimen'].value_counts())

# Persentase
print("\nPersentase Sentimen:")
print(df['sentimen'].value_counts(normalize=True) * 100)

# Panjang review
df['review_length'] = df['review'].str.len()
df['word_count'] = df['review'].str.split().str.len()

print("\n\nStatistik Panjang Review (karakter):")
print(f"  Min: {df['review_length'].min()}")
print(f"  Max: {df['review_length'].max()}")
print(f"  Mean: {df['review_length'].mean():.2f}")
print(f"  Median: {df['review_length'].median():.2f}")
print(f"  Std Dev: {df['review_length'].std():.2f}")

print("\nStatistik Jumlah Kata:")
print(f"  Min: {df['word_count'].min()}")
print(f"  Max: {df['word_count'].max()}")
print(f"  Mean: {df['word_count'].mean():.2f}")
print(f"  Median: {df['word_count'].median():.2f}")

# Perbandingan per sentimen
print("\n\nPanjang Review Rata-rata per Sentimen:")
print(df.groupby('sentimen')['review_length'].agg(['min', 'max', 'mean', 'median']))

# ============================================================================
# CONTOH 3: VISUALISASI DATA
# ============================================================================
print("\n\n[CONTOH 3] Membuat Visualisasi Data")
print("-" * 70)

# Inisialisasi visualizer
visualizer = SentimenVisualizer(df)

# Buat overview grafik
print("\nMembuat overview grafik (4 subplot)...")
visualizer.plot_overview(save_path='visualisasi_sentimen.png')

# Buat review length comparison
print("Membuat review length comparison...")
visualizer.plot_review_length_comparison(save_path='review_length_comparison.png')

# ============================================================================
# CONTOH 4: PERSIAPAN DATA UNTUK ML
# ============================================================================
print("\n\n[CONTOH 4] Persiapan Data untuk Machine Learning")
print("-" * 70)

# Separasi features dan target
X = df['review']
y = df['sentimen']

# Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nData Split:")
print(f"  Training set: {len(X_train)} sampel ({len(X_train)/len(df)*100:.1f}%)")
print(f"  Testing set: {len(X_test)} sampel ({len(X_test)/len(df)*100:.1f}%)")

print(f"\nDistribusi Training Set:")
print(f"  Positif: {sum(y_train == 'Positif')} ({sum(y_train == 'Positif')/len(y_train)*100:.1f}%)")
print(f"  Negatif: {sum(y_train == 'Negatif')} ({sum(y_train == 'Negatif')/len(y_train)*100:.1f}%)")

print(f"\nDistribusi Testing Set:")
print(f"  Positif: {sum(y_test == 'Positif')} ({sum(y_test == 'Positif')/len(y_test)*100:.1f}%)")
print(f"  Negatif: {sum(y_test == 'Negatif')} ({sum(y_test == 'Negatif')/len(y_test)*100:.1f}%)")

# ============================================================================
# CONTOH 5: FEATURE EXTRACTION DENGAN TF-IDF
# ============================================================================
print("\n\n[CONTOH 5] Feature Extraction dengan TF-IDF")
print("-" * 70)

# Vectorizer
vectorizer = TfidfVectorizer(
    max_features=100,
    lowercase=True,
    stop_words='english',
    min_df=1,
    max_df=0.8
)

# Fit dan transform
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"\nTF-IDF Vectorization Results:")
print(f"  Dimensi training set: {X_train_tfidf.shape}")
print(f"  Dimensi testing set: {X_test_tfidf.shape}")
print(f"  Total features: {X_train_tfidf.shape[1]}")

# Tampilkan sample features
feature_names = vectorizer.get_feature_names_out()
print(f"\nSample Features (kata-kata):")
print(f"  {', '.join(feature_names[:20])}")

# ============================================================================
# CONTOH 6: TRAINING MODEL NAIVE BAYES
# ============================================================================
print("\n\n[CONTOH 6] Training Model Naive Bayes")
print("-" * 70)

# Training
print("\nMengtraining model Naive Bayes...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
print("✓ Model berhasil dilatih!")

# Model parameters
print(f"\nModel Parameters:")
print(f"  Classes: {model.classes_}")
print(f"  Feature count: {model.n_features_in_}")

# ============================================================================
# CONTOH 7: EVALUASI MODEL
# ============================================================================
print("\n\n[CONTOH 7] Evaluasi Model pada Test Set")
print("-" * 70)

# Prediksi
y_pred = model.predict(X_test_tfidf)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Detailed Classification Report
print("\n\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negatif', 'Positif']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"              Predicted")
print(f"           Negatif  Positif")
print(f"Actual Negatif  {cm[0][0]:>3}      {cm[0][1]:>3}")
print(f"       Positif   {cm[1][0]:>3}      {cm[1][1]:>3}")

# Visualisasi confusion matrix
visualizer.plot_confusion_matrix(cm, save_path='confusion_matrix.png')

# ============================================================================
# CONTOH 8: ANALISIS TOP FEATURES
# ============================================================================
print("\n\n[CONTOH 8] Analisis Top Features")
print("-" * 70)

# Feature importance
feature_importance = model.feature_log_prob_

# Top features untuk setiap kelas
positif_features_idx = np.argsort(feature_importance[1])[-10:][::-1]
negatif_features_idx = np.argsort(feature_importance[0])[-10:][::-1]

print("\nTop 10 Fitur POSITIF:")
for i, idx in enumerate(positif_features_idx, 1):
    print(f"  {i:2d}. {feature_names[idx]:15s} (score: {feature_importance[1][idx]:7.4f})")

print("\nTop 10 Fitur NEGATIF:")
for i, idx in enumerate(negatif_features_idx, 1):
    print(f"  {i:2d}. {feature_names[idx]:15s} (score: {feature_importance[0][idx]:7.4f})")

# Visualisasi top features
visualizer.plot_top_features(feature_names, feature_importance, 
                            save_path='top_features.png', n_features=10)

# ============================================================================
# CONTOH 9: PREDIKSI DENGAN REVIEW CUSTOM
# ============================================================================
print("\n\n[CONTOH 9] Prediksi dengan Review Custom")
print("-" * 70)

custom_reviews = [
    "Aplikasi ini sangat bagus dan mudah digunakan, fitur-fiturnya keren",
    "Banyak bug dan sering crash, sangat mengecewakan",
    "Loading lambat dan terlalu banyak iklan",
    "Fitur duet sangat menyenangkan dan kreatif",
    "Tidak bisa menghapus akun dengan mudah, privacy policy membingungkan",
    "Interface user-friendly dan algoritma rekomendasi sangat akurat"
]

print("\nPrediksi Custom Reviews:")
print("=" * 70)

for i, review in enumerate(custom_reviews, 1):
    # Transform
    review_tfidf = vectorizer.transform([review])
    
    # Prediksi
    prediction = model.predict(review_tfidf)[0]
    probabilities = model.predict_proba(review_tfidf)[0]
    
    # Confidence
    confidence = max(probabilities)
    
    # Display
    print(f"\n{i}. Review: \"{review}\"")
    print(f"   Prediksi: {prediction}")
    print(f"   Confidence: {confidence*100:.2f}%")
    print(f"   Probabilitas Negatif: {probabilities[0]*100:.2f}%")
    print(f"   Probabilitas Positif: {probabilities[1]*100:.2f}%")

# ============================================================================
# CONTOH 10: PREDIKSI BATCH
# ============================================================================
print("\n\n[CONTOH 10] Prediksi Batch")
print("-" * 70)

# Prediksi semua test set
print("\nMemprediksi semua test set...")
y_pred_all = model.predict(X_test_tfidf)
y_pred_proba = model.predict_proba(X_test_tfidf)

# Buat dataframe hasil
results_df = pd.DataFrame({
    'review': X_test.values,
    'actual': y_test.values,
    'predicted': y_pred_all,
    'confidence': y_pred_proba.max(axis=1),
    'prob_negatif': y_pred_proba[:, 0],
    'prob_positif': y_pred_proba[:, 1]
})

# Tampilkan hasil
print("\nSample Hasil Prediksi (5 sampel pertama):")
print(results_df.head().to_string())

# Analisis hasil
correct = (results_df['actual'] == results_df['predicted']).sum()
incorrect = (results_df['actual'] != results_df['predicted']).sum()

print(f"\n\nRingkasan Hasil Prediksi:")
print(f"  Total prediksi: {len(results_df)}")
print(f"  Benar: {correct} ({correct/len(results_df)*100:.1f}%)")
print(f"  Salah: {incorrect} ({incorrect/len(results_df)*100:.1f}%)")

# False positives dan false negatives
fp = results_df[(results_df['actual'] == 'Negatif') & (results_df['predicted'] == 'Positif')]
fn = results_df[(results_df['actual'] == 'Positif') & (results_df['predicted'] == 'Negatif')]

print(f"  False Positives: {len(fp)}")
print(f"  False Negatives: {len(fn)}")

# ============================================================================
# CONTOH 11: ERROR ANALYSIS
# ============================================================================
print("\n\n[CONTOH 11] Error Analysis")
print("-" * 70)

# Review yang salah diprediksi
wrong_predictions = results_df[results_df['actual'] != results_df['predicted']]

if len(wrong_predictions) > 0:
    print(f"\nTotal Mispredictions: {len(wrong_predictions)}")
    print("\nContoh Mispredictions (sampai 5):")
    
    for idx, row in wrong_predictions.head(5).iterrows():
        print(f"\n  Review: \"{row['review'][:60]}...\"")
        print(f"  Actual: {row['actual']} | Predicted: {row['predicted']}")
        print(f"  Confidence: {row['confidence']*100:.2f}%")
else:
    print("\n✓ Semua prediksi benar! (tidak ada error)")

# ============================================================================
# CONTOH 12: CROSS VALIDATION
# ============================================================================
print("\n\n[CONTOH 12] Cross Validation (K-Fold)")
print("-" * 70)

from sklearn.model_selection import cross_val_score

# K-Fold Cross Validation
cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)

print(f"\n5-Fold Cross Validation Scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.4f}")

print(f"\nMean CV Score: {cv_scores.mean():.4f}")
print(f"Std Dev: {cv_scores.std():.4f}")
print(f"Range: {cv_scores.min():.4f} - {cv_scores.max():.4f}")

# ============================================================================
# CONTOH 13: EXPORT HASIL ANALISIS
# ============================================================================
print("\n\n[CONTOH 13] Export Hasil Analisis")
print("-" * 70)

# Export hasil prediksi
results_df.to_csv('hasil_prediksi.csv', index=False)
print("✓ Hasil prediksi diexport ke: hasil_prediksi.csv")

# Export top features
top_features_df = pd.DataFrame({
    'feature': feature_names,
    'positif_score': feature_importance[1],
    'negatif_score': feature_importance[0]
})
top_features_df = top_features_df.sort_values('positif_score', ascending=False)
top_features_df.to_csv('features_importance.csv', index=False)
print("✓ Feature importance diexport ke: features_importance.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

summary_text = f"""
Dataset:
  - Total Sampel: {len(df)}
  - Positif: {sum(df['sentimen'] == 'Positif')} ({sum(df['sentimen'] == 'Positif')/len(df)*100:.1f}%)
  - Negatif: {sum(df['sentimen'] == 'Negatif')} ({sum(df['sentimen'] == 'Negatif')/len(df)*100:.1f}%)

Model:
  - Algorithm: Multinomial Naive Bayes
  - Features: TF-IDF (max 100)
  - Training Samples: {len(X_train)}
  - Testing Samples: {len(X_test)}

Performance:
  - Accuracy: {accuracy*100:.2f}%
  - Mean CV Score: {cv_scores.mean()*100:.2f}%
  - Standard Deviation: {cv_scores.std()*100:.2f}%

Top Insight:
  - Review Positif Paling Menekankan: {feature_names[positif_features_idx[0]]}
  - Review Negatif Paling Menekankan: {feature_names[negatif_features_idx[0]]}
"""

print(summary_text)

print("\n✅ ANALISIS SELESAI!")
print("\nFile yang telah dibuat:")
print("  1. visualisasi_sentimen.png")
print("  2. review_length_comparison.png")
print("  3. confusion_matrix.png")
print("  4. top_features.png")
print("  5. hasil_prediksi.csv")
print("  6. features_importance.csv")

print("\n" + "=" * 70)
