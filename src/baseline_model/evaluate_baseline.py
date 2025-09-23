import os
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

CSV_PATH = "processed_data/metadata_features.csv"
MODELS_DIR = "models"
RESULTS_DIR = "results"

def evaluate_model(model_file, name):
    df = pd.read_csv(CSV_PATH)
    drop_cols = [c for c in ["file_name", "main_class", "resolution", "class_label"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["class_label"] if "class_label" in df.columns else df["main_class"]

    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    model = joblib.load(model_file)

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    print(f"\n=== {name} Evaluation ===")
    print("Accuracy:", accuracy_score(y, y_pred))
    print("Macro F1:", f1_score(y, y_pred, average="macro"))
    print(classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_)
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    out = os.path.join(RESULTS_DIR, f"{name.replace(' ','_')}_cm.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved", out)

if __name__ == "__main__":
    evaluate_model(os.path.join(MODELS_DIR, "random_forest.pkl"), "Random Forest")
    evaluate_model(os.path.join(MODELS_DIR, "svm.pkl"), "SVM")
