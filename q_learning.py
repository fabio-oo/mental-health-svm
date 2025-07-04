from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

def clean_and_train_svm(df, target_column):
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=[target_column])
    df = df.fillna(df.median(numeric_only=True))

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode fitur kategorikal
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Encode label jika perlu
    if y.dtype == 'O':
        y = LabelEncoder().fit_transform(y.astype(str))

    # Tambah kompleksitas dengan fitur polinomial
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Normalisasi
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    # Bagi data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # Model SVM — konfigurasi kompleks (overfitting rawan)
    model = SVC(kernel='linear', C=1)
    model.fit(X_train, y_train)

    # Evaluasi training
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)

    # Evaluasi testing
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    report = classification_report(y_test, y_test_pred, output_dict=True)
    cmatrix = confusion_matrix(y_test, y_test_pred)

    return {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "report": report,
        "confusion_matrix": cmatrix
    }
