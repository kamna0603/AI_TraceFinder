import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

CSV_PATH = "processed_data/metadata_features.csv"
MODELS_DIR = "models"

def train_models():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(CSV_PATH)

    # Drop only existing columns
    drop_cols = [c for c in ["file_name", "main_class", "resolution", "class_label"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    # Pick label column
    if "class_label" in df.columns:
        y = df["class_label"]
    elif "main_class" in df.columns:
        y = df["main_class"]
    else:
        raise ValueError("No class label column found in CSV")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(X_train_s, y_train)
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))

    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True)
    svm.fit(X_train_s, y_train)
    joblib.dump(svm, os.path.join(MODELS_DIR, "svm.pkl"))

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    print("✅ Models trained and saved to", MODELS_DIR)

if __name__ == "__main__":
    train_models()
