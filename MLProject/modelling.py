import pandas as pd
import mlflow
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# -------------------------------
# Ambil parameter dari CLI
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--max_depth', type=int, default=5)
args = parser.parse_args()

# -------------------------------
# Load dataset hasil preprocessing
# -------------------------------
X_train = pd.read_csv('obesity_preprocessing/X_train.csv').astype('float')
X_test = pd.read_csv('obesity_preprocessing/X_test.csv').astype('float')
y_train = pd.read_csv('obesity_preprocessing/y_train.csv').squeeze()
y_test = pd.read_csv('obesity_preprocessing/y_test.csv').squeeze()

# -------------------------------
# MLflow manual tracking
# -------------------------------

with mlflow.start_run(nested=True): 
    # Logging parameter secara manual
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    # Training model
    clf = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth)
    clf.fit(X_train, y_train)

    # Prediksi dan evaluasi
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')

    # Logging metrics secara manual
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    print("Akurasi:", acc)

    # Simpan model sebagai artifact
    mlflow.sklearn.log_model(clf, "model")
