"""
Modul Visualisasi Grafik untuk Analisis Sentimen TikTok
Menyediakan fungsi-fungsi untuk membuat berbagai jenis visualisasi data
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec

# Konfigurasi global
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
sns.set_style("whitegrid")

class SentimenVisualizer:
    """Kelas untuk membuat berbagai visualisasi data sentimen"""
    
    def __init__(self, df, figsize=(14, 10)):
        """
        Inisialisasi visualizer
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame dengan kolom 'review' dan 'sentimen'
        figsize : tuple
            Ukuran figure (width, height)
        """
        self.df = df
        self.figsize = figsize
        self.sentimen_count = df['sentimen'].value_counts()
        self.colors = {'Positif': '#00D084', 'Negatif': '#FF3B30'}
        
    def plot_overview(self, save_path='visualisasi_sentimen.png'):
        """
        Buat overview grafik lengkap (4 subplot)
        
        Parameters:
        -----------
        save_path : str
            Path untuk menyimpan hasil visualisasi
        """
        fig = plt.figure(figsize=self.figsize)
        gs = GridSpec(2, 2, figure=fig)
        
        fig.suptitle('Analisis Sentimen Review Aplikasi TikTok (200 Sampel)', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        # Plot 1: Bar Chart
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_bar_chart(ax1)
        
        # Plot 2: Pie Chart
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_pie_chart(ax2)
        
        # Plot 3: Histogram
        ax3 = fig.add_subplot(gs[1, 0])
        self._plot_histogram(ax3)
        
        # Plot 4: Statistics
        ax4 = fig.add_subplot(gs[1, 1])
        self._plot_statistics(ax4)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualisasi disimpan: {save_path}")
        plt.show()
        
    def _plot_bar_chart(self, ax):
        """Plot bar chart distribusi sentimen"""
        self.sentimen_count.plot(kind='bar', ax=ax, 
                                 color=[self.colors['Positif'], self.colors['Negatif']])
        ax.set_title('Distribusi Total Sentimen', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sentimen', fontsize=10)
        ax.set_ylabel('Jumlah', fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        
        # Tambah nilai di atas bar
        for i, v in enumerate(self.sentimen_count):
            ax.text(i, v + 1, str(v), ha='center', fontweight='bold', fontsize=11)
        
        ax.grid(axis='y', alpha=0.3)
        
    def _plot_pie_chart(self, ax):
        """Plot pie chart proporsi sentimen"""
        colors_list = [self.colors['Positif'], self.colors['Negatif']]
        wedges, texts, autotexts = ax.pie(
            self.sentimen_count, 
            labels=self.sentimen_count.index,
            autopct='%1.1f%%', 
            colors=colors_list,
            startangle=90,
            textprops={'fontsize': 10, 'fontweight': 'bold'}
        )
        
        # Styling
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        ax.set_title('Proporsi Sentimen (%)', fontsize=12, fontweight='bold')
        
    def _plot_histogram(self, ax):
        """Plot histogram panjang review"""
        self.df['review_length'] = self.df['review'].str.len()
        
        sentimen_positif = self.df[self.df['sentimen'] == 'Positif']['review_length']
        sentimen_negatif = self.df[self.df['sentimen'] == 'Negatif']['review_length']
        
        ax.hist([sentimen_positif, sentimen_negatif], 
                bins=20, 
                label=['Positif', 'Negatif'],
                color=[self.colors['Positif'], self.colors['Negatif']],
                alpha=0.7,
                edgecolor='black')
        
        ax.set_title('Distribusi Panjang Review per Sentimen', fontsize=12, fontweight='bold')
        ax.set_xlabel('Panjang Review (karakter)', fontsize=10)
        ax.set_ylabel('Frekuensi', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
    def _plot_statistics(self, ax):
        """Plot statistik deskriptif"""
        ax.axis('off')
        
        self.df['review_length'] = self.df['review'].str.len()
        sentimen_positif = self.df[self.df['sentimen'] == 'Positif']['review_length']
        sentimen_negatif = self.df[self.df['sentimen'] == 'Negatif']['review_length']
        
        stats_text = f"""
STATISTIK DESKRIPTIF:

Total Sampel: {len(self.df)}
Sentimen Positif: {self.sentimen_count['Positif']} ({self.sentimen_count['Positif']/len(self.df)*100:.1f}%)
Sentimen Negatif: {self.sentimen_count['Negatif']} ({self.sentimen_count['Negatif']/len(self.df)*100:.1f}%)

Panjang Review (karakter):
  Min: {self.df['review_length'].min()}
  Max: {self.df['review_length'].max()}
  Rata-rata: {self.df['review_length'].mean():.1f}
  Median: {self.df['review_length'].median():.1f}

Panjang Review Positif:
  Rata-rata: {sentimen_positif.mean():.1f}
  Median: {sentimen_positif.median():.1f}

Panjang Review Negatif:
  Rata-rata: {sentimen_negatif.mean():.1f}
  Median: {sentimen_negatif.median():.1f}
"""
        
        ax.text(0.05, 0.5, stats_text, fontsize=9.5, verticalalignment='center',
                fontfamily='monospace', 
                bbox=dict(boxstyle='round', facecolor='#fffacd', alpha=0.4, pad=1))
        
    def plot_confusion_matrix(self, cm, save_path='confusion_matrix.png'):
        """
        Plot confusion matrix
        
        Parameters:
        -----------
        cm : numpy.ndarray
            Confusion matrix
        save_path : str
            Path untuk menyimpan hasil
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Negatif', 'Positif'],
                   yticklabels=['Negatif', 'Positif'],
                   cbar_kws={'label': 'Jumlah'},
                   annot_kws={'fontsize': 14, 'fontweight': 'bold'})
        
        ax.set_title('Confusion Matrix - Klasifikasi Sentimen TikTok', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion Matrix disimpan: {save_path}")
        plt.show()
        
    def plot_top_features(self, feature_names, feature_importance, 
                         save_path='top_features.png', n_features=10):
        """
        Plot top features untuk setiap kelas
        
        Parameters:
        -----------
        feature_names : array
            Nama-nama fitur
        feature_importance : array
            Log probability untuk setiap fitur dan kelas
        save_path : str
            Path untuk menyimpan hasil
        n_features : int
            Jumlah top features yang ditampilkan
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Top Positif Features
        positif_features = np.argsort(feature_importance[1])[-n_features:][::-1]
        top_pos_words = [feature_names[i] for i in positif_features]
        top_pos_scores = [feature_importance[1][i] for i in positif_features]
        
        axes[0].barh(top_pos_words, top_pos_scores, color=self.colors['Positif'], 
                    edgecolor='black', linewidth=1.2)
        axes[0].set_title(f'Top {n_features} Fitur Sentimen POSITIF', 
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Log Probability', fontsize=10, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(axis='x', alpha=0.3)
        
        # Top Negatif Features
        negatif_features = np.argsort(feature_importance[0])[-n_features:][::-1]
        top_neg_words = [feature_names[i] for i in negatif_features]
        top_neg_scores = [feature_importance[0][i] for i in negatif_features]
        
        axes[1].barh(top_neg_words, top_neg_scores, color=self.colors['Negatif'],
                    edgecolor='black', linewidth=1.2)
        axes[1].set_title(f'Top {n_features} Fitur Sentimen NEGATIF', 
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Log Probability', fontsize=10, fontweight='bold')
        axes[1].invert_yaxis()
        axes[1].grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Top Features disimpan: {save_path}")
        plt.show()
        
    def plot_model_performance(self, report_dict, save_path='model_performance.png'):
        """
        Plot performa model (precision, recall, f1-score)
        
        Parameters:
        -----------
        report_dict : dict
            Classification report dalam bentuk dict
        save_path : str
            Path untuk menyimpan hasil
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Model Performance Metrics', fontsize=14, fontweight='bold')
        
        metrics = ['precision', 'recall', 'f1-score']
        classes = ['Negatif', 'Positif']
        
        for idx, metric in enumerate(metrics):
            values = [report_dict['Negatif'][metric], report_dict['Positif'][metric]]
            colors_list = [self.colors['Negatif'], self.colors['Positif']]
            
            bars = axes[idx].bar(classes, values, color=colors_list, 
                                edgecolor='black', linewidth=1.5, alpha=0.8)
            axes[idx].set_title(f'{metric.capitalize()}', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel('Score', fontsize=10)
            axes[idx].set_ylim([0, 1])
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Tambah nilai di atas bar
            for bar in bars:
                height = bar.get_height()
                axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.3f}', ha='center', va='bottom',
                             fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Model Performance disimpan: {save_path}")
        plt.show()
        
    def plot_review_length_comparison(self, save_path='review_length_comparison.png'):
        """
        Plot perbandingan panjang review antara sentimen positif dan negatif
        
        Parameters:
        -----------
        save_path : str
            Path untuk menyimpan hasil
        """
        self.df['review_length'] = self.df['review'].str.len()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Box plot
        sentimen_data = [
            self.df[self.df['sentimen'] == 'Positif']['review_length'],
            self.df[self.df['sentimen'] == 'Negatif']['review_length']
        ]
        
        bp = axes[0].boxplot(sentimen_data, labels=['Positif', 'Negatif'],
                            patch_artist=True, widths=0.6)
        
        for patch, color in zip(bp['boxes'], [self.colors['Positif'], self.colors['Negatif']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[0].set_title('Box Plot Panjang Review', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Panjang (karakter)', fontsize=10)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Violin plot
        parts = axes[1].violinplot(sentimen_data, positions=[1, 2], 
                                   showmeans=True, showmedians=True)
        
        axes[1].set_xticks([1, 2])
        axes[1].set_xticklabels(['Positif', 'Negatif'])
        axes[1].set_title('Violin Plot Panjang Review', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Panjang (karakter)', fontsize=10)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Review Length Comparison disimpan: {save_path}")
        plt.show()
