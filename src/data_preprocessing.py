"""
Data Preprocessing Module
=========================
Modul untuk memuat, membersihkan, dan mempersiapkan data
sebelum digunakan untuk pelatihan model klasifikasi.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path: str) -> pd.DataFrame:
    """
    Memuat dataset dari file CSV.

    Parameters
    ----------
    path : str
        Path relatif ke file CSV dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame berisi seluruh data dari CSV.
    """
    return pd.read_csv(path)


def preprocess_data(df: pd.DataFrame):
    """
    Memisahkan fitur (X) dan label (y) dari DataFrame.

    Label di-encode menjadi numerik:
    - orange      → 0
    - grapefruit  → 1

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame mentah dari dataset.

    Returns
    -------
    X : pd.DataFrame
        Fitur-fitur (diameter, weight, red, green, blue).
    y : pd.Series
        Label yang sudah di-encode (0 atau 1).
    """
    X = df.drop("name", axis=1)
    y = df["name"].map({"orange": 0, "grapefruit": 1})
    return X, y


def split_and_scale(X, y, test_size=0.2, random_state=42):
    """
    Membagi data menjadi train/test set dan melakukan feature scaling.

    Menggunakan StandardScaler untuk normalisasi fitur agar semua fitur
    memiliki mean=0 dan std=1. Ini penting terutama untuk SVM.

    Parameters
    ----------
    X : array-like
        Fitur-fitur dataset.
    y : array-like
        Label dataset.
    test_size : float, default=0.2
        Proporsi data untuk test set (20%).
    random_state : int, default=42
        Seed untuk reproducibility.

    Returns
    -------
    X_train : np.ndarray
        Fitur training yang sudah di-scale.
    X_test : np.ndarray
        Fitur testing yang sudah di-scale.
    y_train : pd.Series
        Label training.
    y_test : pd.Series
        Label testing.
    scaler : StandardScaler
        Objek scaler yang sudah di-fit (untuk disimpan/reuse).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler