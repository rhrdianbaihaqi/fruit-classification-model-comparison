"""
Utility Module
==============
Modul utilitas untuk menyimpan model, scaler, hasil evaluasi,
dan membuat visualisasi perbandingan model.
"""

import os
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend untuk server/CI
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def save_model(model, name, output_dir="models"):
    """
    Menyimpan model yang sudah dilatih ke file .pkl.

    Parameters
    ----------
    model : estimator
        Model yang sudah dilatih.
    name : str
        Nama file model (tanpa ekstensi).
    output_dir : str, default="models"
        Direktori untuk menyimpan model.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  ✅ Model disimpan: {path}")


def save_scaler(scaler, output_dir="models"):
    """
    Menyimpan StandardScaler untuk reuse saat prediksi data baru.

    Parameters
    ----------
    scaler : StandardScaler
        Scaler yang sudah di-fit pada data training.
    output_dir : str, default="models"
        Direktori untuk menyimpan scaler.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "scaler.pkl")
    joblib.dump(scaler, path)
    print(f"  ✅ Scaler disimpan: {path}")


def save_results(results, output_dir="results"):
    """
    Menyimpan hasil evaluasi semua model ke file teks.

    Parameters
    ----------
    results : dict
        Dictionary {model_name: (accuracy, report, cm, y_pred)}.
    output_dir : str, default="results"
        Direktori output.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "metrics.txt")

    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("HASIL EVALUASI MODEL KLASIFIKASI BUAH\n")
        f.write("=" * 60 + "\n\n")

        for model_name, (acc, report, cm, _) in results.items():
            f.write(f"{'─' * 40}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"{'─' * 40}\n")
            f.write(f"Accuracy: {acc:.4f} ({acc * 100:.2f}%)\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write(f"\nConfusion Matrix:\n{cm}\n")
            f.write("\n\n")

    print(f"  ✅ Metrics disimpan: {path}")


def plot_confusion_matrices(results, output_dir="results"):
    """
    Membuat plot confusion matrix untuk setiap model.

    Parameters
    ----------
    results : dict
        Dictionary {model_name: (accuracy, report, cm, y_pred)}.
    output_dir : str, default="results"
        Direktori output.
    """
    os.makedirs(output_dir, exist_ok=True)
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))

    if n_models == 1:
        axes = [axes]

    class_names = ["Orange", "Grapefruit"]

    for ax, (name, (acc, _, cm, __)) in zip(axes, results.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax,
        )
        ax.set_title(f"{name}\nAccuracy: {acc:.4f}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Confusion matrices disimpan: {path}")


def plot_accuracy_comparison(results, output_dir="results"):
    """
    Membuat bar chart perbandingan akurasi antar model.

    Parameters
    ----------
    results : dict
        Dictionary {model_name: (accuracy, report, cm, y_pred)}.
    output_dir : str, default="results"
        Direktori output.
    """
    os.makedirs(output_dir, exist_ok=True)

    names = list(results.keys())
    accuracies = [results[n][0] for n in names]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, accuracies, color=colors[:len(names)], edgecolor="white")

    # Tambah label di atas bar
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{acc:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Perbandingan Akurasi Model Klasifikasi", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "accuracy_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Accuracy comparison disimpan: {path}")


def plot_feature_distribution(df, output_dir="results"):
    """
    Membuat histogram distribusi setiap fitur, dipisahkan per kelas.
    Berguna untuk melihat fitur mana yang paling membedakan kedua buah.
    """
    os.makedirs(output_dir, exist_ok=True)
    features = [col for col in df.columns if col != "name"]
    n = len(features)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    colors = {"orange": "#FF9800", "grapefruit": "#E91E63"}

    for ax, feat in zip(axes, features):
        for label, color in colors.items():
            subset = df[df["name"] == label][feat]
            ax.hist(subset, bins=30, alpha=0.6, label=label, color=color, edgecolor="white")
        ax.set_title(feat, fontsize=12, fontweight="bold")
        ax.set_xlabel(feat)
        ax.set_ylabel("Frekuensi")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Distribusi Fitur per Kelas", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "feature_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Feature distribution disimpan: {path}")


def plot_correlation_heatmap(df, output_dir="results"):
    """
    Membuat heatmap korelasi antar fitur numerik.
    Berguna untuk melihat apakah ada fitur yang redundan (korelasi tinggi).
    """
    os.makedirs(output_dir, exist_ok=True)
    numeric_df = df.select_dtypes(include=[np.number])

    fig, ax = plt.subplots(figsize=(7, 5))
    corr = numeric_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        ax=ax,
        vmin=-1,
        vmax=1,
    )
    ax.set_title("Correlation Heatmap antar Fitur", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Correlation heatmap disimpan: {path}")


def plot_feature_importance(dt_model, feature_names, output_dir="results"):
    """
    Membuat bar chart feature importance dari Decision Tree.
    Menunjukkan fitur mana yang paling berpengaruh dalam klasifikasi.
    """
    os.makedirs(output_dir, exist_ok=True)

    importances = dt_model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_names)))

    bars = ax.bar(
        range(len(feature_names)),
        importances[indices],
        color=colors,
        edgecolor="white",
    )

    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in indices], fontsize=11)
    ax.set_ylabel("Importance", fontsize=12)
    ax.set_title("Feature Importance (Decision Tree)", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Label di atas bar
    for bar, imp in zip(bars, importances[indices]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{imp:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Feature importance disimpan: {path}")


def plot_metrics_comparison(results, output_dir="results"):
    """
    Membuat grouped bar chart perbandingan Precision, Recall, F1-Score
    untuk setiap model (macro average).
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    os.makedirs(output_dir, exist_ok=True)

    model_names = list(results.keys())
    metrics_data = {"Precision": [], "Recall": [], "F1-Score": []}

    for name in model_names:
        _, _, _, y_pred = results[name]
        # Ambil y_test dari confusion matrix (reconstruct)
        cm = results[name][2]
        # Gunakan macro average dari classification report
        # Parse dari report string
        report = results[name][1]
        lines = report.strip().split("\n")
        # Cari baris "macro avg"
        for line in lines:
            if "macro avg" in line:
                parts = line.split()
                idx = parts.index("avg") + 1
                metrics_data["Precision"].append(float(parts[idx]))
                metrics_data["Recall"].append(float(parts[idx + 1]))
                metrics_data["F1-Score"].append(float(parts[idx + 2]))
                break

    x = np.arange(len(model_names))
    width = 0.25
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (metric, values) in enumerate(metrics_data.items()):
        bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i], edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_title("Perbandingan Precision, Recall, F1-Score per Model", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(output_dir, "metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Metrics comparison disimpan: {path}")


def plot_pairplot(df, output_dir="results"):
    """
    Membuat pairplot (scatter matrix) semua kombinasi fitur.
    Berguna untuk melihat separasi antar kelas di berbagai fitur.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Sample data untuk kecepatan (pairplot lambat dengan 10k rows)
    sample_df = df.sample(n=min(2000, len(df)), random_state=42)

    palette = {"orange": "#FF9800", "grapefruit": "#E91E63"}
    g = sns.pairplot(
        sample_df,
        hue="name",
        palette=palette,
        diag_kind="hist",
        plot_kws={"alpha": 0.5, "s": 15},
        diag_kws={"alpha": 0.6},
    )
    g.figure.suptitle("Pairplot — Semua Kombinasi Fitur", y=1.01, fontsize=14, fontweight="bold")

    path = os.path.join(output_dir, "pairplot.png")
    g.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Pairplot disimpan: {path}")