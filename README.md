# 🍊 Fruit Classification — Model Comparison

Proyek ini membandingkan tiga model klasifikasi machine learning untuk mengidentifikasi buah **Orange** vs **Grapefruit** berdasarkan fitur fisik dan warna.

## 👤 Penulis

| | |
|---|---|
| **Nama** | Muhammad Rahardian Baihaqi |
| **NIM** | 1237050023 |
| **Mata Kuliah** | Machine Learning |
| **Tugas** | Ujian Tengah Semester (UTS) |

---

## 📝 Deskripsi

Dataset `citrus.csv` berisi 10.000 data buah (5.000 orange, 5.000 grapefruit) dengan fitur:

| Fitur | Deskripsi |
|-------|-----------|
| `diameter` | Diameter buah (cm) |
| `weight` | Berat buah (gram) |
| `red` | Nilai warna merah (0-255) |
| `green` | Nilai warna hijau (0-255) |
| `blue` | Nilai warna biru (0-255) |

---

## 🧪 Tahapan Pembuatan Model (Step-by-Step)

### Step 1 — Load Data

Memuat dataset `citrus.csv` menggunakan `pandas`.

```python
df = pd.read_csv("data/citrus.csv")
# Output: (10000, 6) — 10.000 baris, 6 kolom
```

- Dataset memiliki **distribusi seimbang** (balanced): 5.000 orange, 5.000 grapefruit.
- Tidak ada missing values.

### Step 2 — Preprocessing

#### a. Pemisahan Fitur dan Label

```python
X = df.drop("name", axis=1)              # 5 fitur numerik
y = df["name"].map({"orange": 0, "grapefruit": 1})  # Encode label
```

#### b. Train-Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Training: 8.000 samples (4.000 orange + 4.000 grapefruit)
# Testing:  2.000 samples (1.000 orange + 1.000 grapefruit)
```

Menggunakan rasio **80:20** dengan:
- `random_state=42` → menjamin **reproducibility** (hasil konsisten setiap dijalankan)
- `stratify=y` → menjamin **distribusi label proporsional** (50:50) di train dan test set

#### c. Feature Scaling (StandardScaler)

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # fit + transform pada training
X_test = scaler.transform(X_test)        # hanya transform pada testing
```

Scaling mengubah setiap fitur agar memiliki **mean=0** dan **std=1**. Ini krusial untuk SVM yang sensitif terhadap skala fitur.

### Step 3 — Training Model

#### a. Decision Tree

```python
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
```

- **Cara kerja**: Membagi data secara rekursif berdasarkan fitur yang paling informatif (Gini impurity).
- **Kelebihan**: Mudah diinterpretasi, bisa menunjukkan feature importance.
- **Kekurangan**: Rentan overfitting jika tidak di-prune.

#### b. Naive Bayes (Gaussian)

```python
model = GaussianNB()
model.fit(X_train, y_train)
```

- **Cara kerja**: Menggunakan teorema Bayes dengan asumsi independensi antar fitur dan distribusi Gaussian.
- **Kelebihan**: Sangat cepat, bekerja baik dengan dataset kecil.
- **Kekurangan**: Asumsi independensi jarang terpenuhi di dunia nyata.

#### c. Support Vector Machine (SVM)

```python
model = SVC(kernel="rbf", random_state=42)
model.fit(X_train, y_train)
```

- **Cara kerja**: Mencari hyperplane optimal yang memisahkan dua kelas dengan margin maksimal.
- **Kelebihan**: Efektif di high-dimensional space, robust terhadap outlier.
- **Kekurangan**: Lambat untuk dataset besar, wajib feature scaling.

### Step 4 — Evaluasi

Setiap model dievaluasi menggunakan metrik berikut:

| Metrik | Penjelasan |
|--------|-----------|
| **Accuracy** | Proporsi prediksi yang benar dari total data |
| **Precision** | Dari semua yang diprediksi positif, berapa yang benar |
| **Recall** | Dari semua data positif, berapa yang berhasil terdeteksi |
| **F1-Score** | Harmonic mean dari precision dan recall |
| **Confusion Matrix** | Tabel 2x2 yang menunjukkan TP, TN, FP, FN |

### Step 5 — Exploratory Data Analysis (EDA)

Visualisasi untuk memahami karakteristik data:

- **Feature Distribution**: Histogram setiap fitur dipisah per kelas
- **Correlation Heatmap**: Korelasi antar fitur numerik
- **Pairplot**: Scatter plot semua kombinasi fitur

### Step 6 — Model Visualization

Visualisasi perbandingan performa model:

- **Confusion Matrix**: Heatmap per model
- **Accuracy Comparison**: Bar chart perbandingan akurasi
- **Precision/Recall/F1 Comparison**: Grouped bar chart
- **Feature Importance**: Fitur paling berpengaruh (dari Decision Tree)

### Step 7 — Penyimpanan

- Model disimpan dalam format `.pkl` di folder `models/`
- Scaler disimpan untuk reuse saat prediksi data baru
- Metrics disimpan di `results/metrics.txt`
- Semua visualisasi disimpan sebagai `.png` di `results/`

---

## 📊 Hasil Evaluasi

### Perbandingan Akurasi

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Decision Tree** | **93.90%** 🏆 | 0.94 | 0.94 | 0.94 |
| SVM | 93.75% | 0.94 | 0.94 | 0.94 |
| Naive Bayes | 92.25% | 0.92 | 0.92 | 0.92 |

### Confusion Matrix

| Model | TP (Orange benar) | FP | FN | TN (Grapefruit benar) |
|-------|---|---|---|---|
| Decision Tree | 943 | 57 | 65 | 935 |
| Naive Bayes | 918 | 82 | 73 | 927 |
| SVM | 931 | 69 | 56 | 944 |

### Visualisasi Output

Semua visualisasi tersimpan di folder `results/`:

| File | Deskripsi |
|------|-----------|
| `accuracy_comparison.png` | Bar chart perbandingan akurasi 3 model |
| `confusion_matrices.png` | Heatmap confusion matrix tiap model |
| `metrics_comparison.png` | Grouped bar chart precision/recall/f1 |
| `feature_importance.png` | Feature importance dari Decision Tree |
| `feature_distribution.png` | Histogram distribusi fitur per kelas |
| `correlation_heatmap.png` | Heatmap korelasi antar fitur |
| `pairplot.png` | Scatter matrix semua kombinasi fitur |
| `metrics.txt` | Classification report lengkap tiap model |

---

## 🏁 Kesimpulan

1. **Decision Tree** menghasilkan akurasi tertinggi (**93.90%**), unggul dalam kemampuan menangkap pola non-linear pada data buah.

2. **SVM** menempati posisi kedua (**93.75%**) dengan keunggulan pada recall kelas grapefruit (944 benar vs 935 Decision Tree), menunjukkan SVM lebih sedikit salah mengklasifikasikan grapefruit.

3. **Naive Bayes** memiliki akurasi terendah (**92.25%**) karena asumsi independensi antar fitur tidak sepenuhnya terpenuhi — beberapa fitur (diameter & weight) memiliki korelasi.

4. **Feature terpenting** berdasarkan Decision Tree: `diameter` dan `weight` merupakan pembeda utama antara orange dan grapefruit (grapefruit umumnya lebih besar dan berat).

5. Secara keseluruhan, **ketiga model memiliki performa yang baik (>92%)** berkat dataset yang seimbang dan penggunaan **stratified sampling** yang menjamin distribusi label proporsional pada train/test set.

---

## 📁 Struktur Folder

```
fruit-classification-model-comparison/
├── README.md                  # Dokumentasi proyek (file ini)
├── .gitignore                 # File yang diabaikan Git
├── requirements.txt           # Daftar dependensi Python
├── main.py                    # Script utama (entry point)
├── data/
│   └── citrus.csv             # Dataset buah (10.000 rows)
├── src/
│   ├── __init__.py            # Package marker
│   ├── data_preprocessing.py  # Load, encode, split, scale data
│   ├── train.py               # Training 3 model klasifikasi
│   ├── evaluate.py            # Evaluasi model (accuracy, CM, dll)
│   └── utils.py               # Save model, scaler, visualisasi
├── models/                    # Output: model & scaler .pkl
│   ├── Decision_Tree.pkl
│   ├── Naive_Bayes.pkl
│   ├── SVM.pkl
│   └── scaler.pkl
└── results/                   # Output: metrics & charts
    ├── metrics.txt
    ├── accuracy_comparison.png
    ├── confusion_matrices.png
    ├── metrics_comparison.png
    ├── feature_importance.png
    ├── feature_distribution.png
    ├── correlation_heatmap.png
    └── pairplot.png
```

---

## 🛠️ Teknologi

| Library | Versi | Kegunaan |
|---------|-------|----------|
| Python | 3.8+ | Bahasa pemrograman |
| pandas | ≥2.0.0 | Manipulasi data |
| numpy | ≥1.24.0 | Operasi numerik |
| scikit-learn | ≥1.3.0 | ML (Decision Tree, Naive Bayes, SVM) |
| matplotlib | ≥3.7.0 | Visualisasi chart |
| seaborn | ≥0.12.0 | Visualisasi statistik |
| joblib | ≥1.3.0 | Serialisasi model (.pkl) |
