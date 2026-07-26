import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# Konfigurasi tampilan
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Load Dataset
print("=" * 60)
print("KLASIFIKASI SENTIMEN APLIKASI TIKTOK")
print("=" * 60)
print("\n[1] Membaca dataset...")
df = pd.read_csv('data.csv')
print(f"Total sampel: {len(df)}")
print(f"\nDataset Info:")
print(df.head(10))

# 2. Analisis Distribusi Sentimen
print("\n[2] Analisis Distribusi Sentimen...")
sentimen_count = df['sentimen'].value_counts()
print(f"\nDistribusi Sentimen:")
print(sentimen_count)
print(f"\nPersentase Sentimen:")
print(df['sentimen'].value_counts(normalize=True) * 100)

# 3. Visualisasi 1: Bar Chart Distribusi Sentimen
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Analisis Sentimen Review Aplikasi TikTok (200 Sampel)', fontsize=16, fontweight='bold')

# Plot 1: Bar Chart
ax1 = axes[0, 0]
sentimen_count.plot(kind='bar', ax=ax1, color=['#00D084', '#FF3B30'])
ax1.set_title('Distribusi Total Sentimen', fontsize=12, fontweight='bold')
ax1.set_xlabel('Sentimen', fontsize=10)
ax1.set_ylabel('Jumlah', fontsize=10)
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
for i, v in enumerate(sentimen_count):
    ax1.text(i, v + 1, str(v), ha='center', fontweight='bold')

# Plot 2: Pie Chart
ax2 = axes[0, 1]
colors = ['#00D084', '#FF3B30']
wedges, texts, autotexts = ax2.pie(sentimen_count, labels=sentimen_count.index, 
                                     autopct='%1.1f%%', colors=colors, startangle=90,
                                     textprops={'fontsize': 10, 'fontweight': 'bold'})
ax2.set_title('Proporsi Sentimen (%)', fontsize=12, fontweight='bold')

# Plot 3: Statistik Panjang Review
ax3 = axes[1, 0]
df['review_length'] = df['review'].str.len()
sentimen_positif = df[df['sentimen'] == 'Positif']['review_length']
sentimen_negatif = df[df['sentimen'] == 'Negatif']['review_length']

ax3.hist([sentimen_positif, sentimen_negatif], bins=20, label=['Positif', 'Negatif'], 
         color=['#00D084', '#FF3B30'], alpha=0.7)
ax3.set_title('Distribusi Panjang Review per Sentimen', fontsize=12, fontweight='bold')
ax3.set_xlabel('Panjang Review (karakter)', fontsize=10)
ax3.set_ylabel('Frekuensi', fontsize=10)
ax3.legend()

# Plot 4: Statistik Deskriptif
ax4 = axes[1, 1]
ax4.axis('off')
stats_text = f"""
STATISTIK DESKRIPTIF:

Total Sampel: {len(df)}
Sentimen Positif: {sentimen_count['Positif']} ({sentimen_count['Positif']/len(df)*100:.1f}%)
Sentimen Negatif: {sentimen_count['Negatif']} ({sentimen_count['Negatif']/len(df)*100:.1f}%)

Panjang Review (karakter):
  Min: {df['review_length'].min()}
  Max: {df['review_length'].max()}
  Rata-rata: {df['review_length'].mean():.1f}
  Median: {df['review_length'].median():.1f}

Panjang Review Positif:
  Rata-rata: {sentimen_positif.mean():.1f}
  Median: {sentimen_positif.median():.1f}

Panjang Review Negatif:
  Rata-rata: {sentimen_negatif.mean():.1f}
  Median: {sentimen_negatif.median():.1f}
"""
ax4.text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
         fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig('visualisasi_sentimen.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualisasi disimpan: visualisasi_sentimen.png")
plt.show()

# 4. Persiapan Data untuk Machine Learning
print("\n[3] Persiapan Data Machine Learning...")
X = df['review']
y = df['sentimen']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {len(X_train)} sampel")
print(f"Testing set: {len(X_test)} sampel")

# 5. TF-IDF Vectorization
print("\n[4] TF-IDF Vectorization...")
vectorizer = TfidfVectorizer(max_features=100, lowercase=True, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
print(f"Fitur yang dihasilkan: {X_train_tfidf.shape[1]}")

# 6. Training Model Naive Bayes
print("\n[5] Training Model Naive Bayes...")
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
print("✓ Model telah dilatih")

# 7. Prediksi dan Evaluasi
print("\n[6] Evaluasi Model...")
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAkurasi Model: {accuracy:.4f} ({accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negatif', 'Positif']))

# 8. Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# 9. Visualisasi Confusion Matrix
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
            xticklabels=['Negatif', 'Positif'], yticklabels=['Negatif', 'Positif'],
            cbar_kws={'label': 'Jumlah'})
ax.set_title('Confusion Matrix - Klasifikasi Sentimen TikTok', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual', fontsize=12)
ax.set_xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
print("\n✓ Confusion Matrix disimpan: confusion_matrix.png")
plt.show()

# 10. Top Features
print("\n[7] Top Features (Kata-kata Penting)...")
feature_names = vectorizer.get_feature_names_out()
feature_importance = model.feature_log_prob_

# Features untuk setiap kelas
positif_features = np.argsort(feature_importance[1])[-10:][::-1]
negatif_features = np.argsort(feature_importance[0])[-10:][::-1]

print("\nTop 10 Fitur Sentimen POSITIF:")
for i, idx in enumerate(positif_features, 1):
    print(f"{i}. {feature_names[idx]}")

print("\nTop 10 Fitur Sentimen NEGATIF:")
for i, idx in enumerate(negatif_features, 1):
    print(f"{i}. {feature_names[idx]}")

# 11. Visualisasi Top Features
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Positif Features
ax1 = axes[0]
top_pos_words = [feature_names[i] for i in positif_features]
top_pos_scores = [feature_importance[1][i] for i in positif_features]
ax1.barh(top_pos_words, top_pos_scores, color='#00D084')
ax1.set_title('Top 10 Fitur Sentimen POSITIF', fontsize=12, fontweight='bold')
ax1.set_xlabel('Log Probability', fontsize=10)
ax1.invert_yaxis()

# Negatif Features
ax2 = axes[1]
top_neg_words = [feature_names[i] for i in negatif_features]
top_neg_scores = [feature_importance[0][i] for i in negatif_features]
ax2.barh(top_neg_words, top_neg_scores, color='#FF3B30')
ax2.set_title('Top 10 Fitur Sentimen NEGATIF', fontsize=12, fontweight='bold')
ax2.set_xlabel('Log Probability', fontsize=10)
ax2.invert_yaxis()

plt.tight_layout()
plt.savefig('top_features.png', dpi=300, bbox_inches='tight')
print("\n✓ Top Features disimpan: top_features.png")
plt.show()

# 12. Testing dengan Custom Reviews
print("\n[8] Testing Model dengan Custom Reviews...")
custom_reviews = [
    "Aplikasi ini sangat bagus dan mudah digunakan",
    "Banyak bug dan sering crash, sangat mengecewakan",
    "Fitur-fitur baru sangat menarik dan kreatif",
    "Terlalu banyak iklan yang mengganggu"
]

print("\nPrediksi Custom Reviews:")
for review in custom_reviews:
    review_tfidf = vectorizer.transform([review])
    prediction = model.predict(review_tfidf)[0]
    probability = model.predict_proba(review_tfidf)[0]
    print(f"\n📝 Review: \"{review}\"")
    print(f"   Prediksi: {prediction}")
    print(f"   Confidence: {max(probability)*100:.2f}%")

print("\n" + "=" * 60)
print("ANALISIS SELESAI")
print("=" * 60)
print("\nFile yang dihasilkan:")
print("  1. visualisasi_sentimen.png - Visualisasi utama sentimen")
print("  2. confusion_matrix.png - Confusion matrix model")
print("  3. top_features.png - Top fitur per sentimen")
