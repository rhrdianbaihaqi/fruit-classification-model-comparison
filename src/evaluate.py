"""
Evaluation Module
=================
Modul untuk mengevaluasi performa model klasifikasi
menggunakan berbagai metrik: accuracy, precision, recall, f1-score,
dan confusion matrix.
"""

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


def evaluate_model(model, X_test, y_test, target_names=None):
    """
    Mengevaluasi performa model pada data test.

    Parameters
    ----------
    model : estimator
        Model yang sudah dilatih.
    X_test : array-like
        Fitur testing yang sudah di-scale.
    y_test : array-like
        Label testing (ground truth).
    target_names : list of str, optional
        Nama kelas untuk classification report.

    Returns
    -------
    accuracy : float
        Akurasi model (0.0 - 1.0).
    report : str
        Classification report (precision, recall, f1-score per kelas).
    cm : np.ndarray
        Confusion matrix 2x2.
    y_pred : np.ndarray
        Prediksi model pada data test.
    """
    if target_names is None:
        target_names = ["Orange", "Grapefruit"]

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, report, cm, y_pred