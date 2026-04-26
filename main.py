"""
Main Script - Fruit Classification Model Comparison
====================================================
Script utama untuk menjalankan seluruh pipeline:
1. Load data
2. Preprocessing (cleaning, encoding, scaling)
3. Training (Decision Tree, Naive Bayes, SVM)
4. Evaluation (accuracy, classification report, confusion matrix)
5. Save models & results (termasuk visualisasi)
"""

from src.data_preprocessing import load_data, preprocess_data, split_and_scale
from src.train import train_decision_tree, train_naive_bayes, train_svm
from src.evaluate import evaluate_model
from src.utils import (
    save_model,
    save_scaler,
    save_results,
    plot_confusion_matrices,
    plot_accuracy_comparison,
    plot_feature_distribution,
    plot_correlation_heatmap,
    plot_feature_importance,
    plot_metrics_comparison,
    plot_pairplot,
)


def main():
    print("=" * 50)
    print("🍊 FRUIT CLASSIFICATION MODEL COMPARISON")
    print("=" * 50)
    print("Pipeline: Load → Preprocess → Train → Evaluate → EDA → Visualize → Save")

    # ──────────────────────────────────────────────
    # 1. Load Data
    # ──────────────────────────────────────────────
    print("\n📂 [1/5] Loading data...")
    df = load_data("data/citrus.csv")
    print(f"  Dataset shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Label distribution:\n{df['name'].value_counts().to_string()}")

    # ──────────────────────────────────────────────
    # 2. Preprocessing
    # ──────────────────────────────────────────────
    print("\n🔧 [2/5] Preprocessing data...")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Testing set:  {X_test.shape[0]} samples")

    # ──────────────────────────────────────────────
    # 3. Training
    # ──────────────────────────────────────────────
    print("\n🏋️ [3/5] Training models...")

    models = {
        "Decision Tree": train_decision_tree(X_train, y_train),
        "Naive Bayes": train_naive_bayes(X_train, y_train),
        "SVM": train_svm(X_train, y_train),
    }

    for name in models:
        print(f"  ✅ {name} trained")

    # ──────────────────────────────────────────────
    # 4. Evaluation
    # ──────────────────────────────────────────────
    print("\n📊 [4/5] Evaluating models...")

    results = {}
    for name, model in models.items():
        acc, report, cm, y_pred = evaluate_model(model, X_test, y_test)
        results[name] = (acc, report, cm, y_pred)
        print(f"\n  ── {name} ──")
        print(f"  Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
        print(f"  Confusion Matrix:\n  {cm}")

    # ──────────────────────────────────────────────
    # 5. EDA Visualizations
    # ──────────────────────────────────────────────
    print("\n📈 [5/7] Generating EDA visualizations...")
    plot_feature_distribution(df)
    plot_correlation_heatmap(df)
    plot_pairplot(df)

    # ──────────────────────────────────────────────
    # 6. Model Visualizations
    # ──────────────────────────────────────────────
    print("\n📊 [6/7] Generating model visualizations...")
    plot_confusion_matrices(results)
    plot_accuracy_comparison(results)
    plot_metrics_comparison(results)

    # Feature importance dari Decision Tree
    feature_names = list(df.drop('name', axis=1).columns)
    plot_feature_importance(models["Decision Tree"], feature_names)

    # ──────────────────────────────────────────────
    # 7. Save Models & Results
    # ──────────────────────────────────────────────
    print("\n💾 [7/7] Saving models & results...")

    for name, model in models.items():
        save_model(model, name.replace(" ", "_"))

    save_scaler(scaler)
    save_results(results)

    # ──────────────────────────────────────────────
    # Summary
    # ──────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("📋 RINGKASAN HASIL")
    print("=" * 50)
    best_name = max(results, key=lambda k: results[k][0])
    for name, (acc, _, __, ___) in results.items():
        marker = " 🏆" if name == best_name else ""
        print(f"  {name:20s} → {acc:.4f}{marker}")
    print(f"\n  Best model: {best_name}")
    print("=" * 50)


if __name__ == "__main__":
    main()