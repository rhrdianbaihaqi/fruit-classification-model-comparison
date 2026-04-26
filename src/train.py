"""
Training Module
===============
Modul untuk melatih tiga model klasifikasi:
1. Decision Tree
2. Naive Bayes (Gaussian)
3. Support Vector Machine (SVM)
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def train_decision_tree(X_train, y_train, random_state=42):
    """
    Melatih model Decision Tree Classifier.

    Decision Tree bekerja dengan membagi data berdasarkan fitur
    yang paling informatif secara rekursif (menggunakan Gini impurity
    atau information gain).

    Parameters
    ----------
    X_train : array-like
        Fitur training yang sudah di-scale.
    y_train : array-like
        Label training.
    random_state : int, default=42
        Seed untuk reproducibility.

    Returns
    -------
    model : DecisionTreeClassifier
        Model yang sudah dilatih.
    """
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_naive_bayes(X_train, y_train):
    """
    Melatih model Gaussian Naive Bayes.

    Naive Bayes menggunakan teorema Bayes dengan asumsi independensi
    antar fitur. Gaussian NB mengasumsikan distribusi normal pada
    setiap fitur.

    Parameters
    ----------
    X_train : array-like
        Fitur training yang sudah di-scale.
    y_train : array-like
        Label training.

    Returns
    -------
    model : GaussianNB
        Model yang sudah dilatih.
    """
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


def train_svm(X_train, y_train, random_state=42):
    """
    Melatih model Support Vector Machine (SVM).

    SVM mencari hyperplane optimal yang memisahkan dua kelas
    dengan margin maksimal. Menggunakan kernel RBF (default).

    Parameters
    ----------
    X_train : array-like
        Fitur training yang sudah di-scale.
    y_train : array-like
        Label training.
    random_state : int, default=42
        Seed untuk reproducibility.

    Returns
    -------
    model : SVC
        Model yang sudah dilatih.
    """
    model = SVC(kernel="rbf", random_state=random_state)
    model.fit(X_train, y_train)
    return model