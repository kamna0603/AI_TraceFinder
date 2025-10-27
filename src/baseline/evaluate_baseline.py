import matplotlib
matplotlib.use('Agg')
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

CSV_PATHS = [
    "processed_data/Flatfield/metadata_features.csv",
    "processed_data/Official/metadata_features.csv"
]

def evaluate_model(model_path, name, save_dir="results"):
    
    df = pd.concat([pd.read_csv(path) for path in CSV_PATHS], ignore_index=True)

    X = df.drop(columns=["file_name", "main_class", "resolution", "class_label"])
    y = df["class_label"]

    scaler = joblib.load("models/scaler.pkl")
    model = joblib.load(model_path)

    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)

    print(f"\n=== {name} Evaluation ===")
    print(classification_report(y, y_pred))

    cm = confusion_matrix(y, y_pred, labels=model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=model.classes_,
                yticklabels=model.classes_,
                cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name.replace(' ', '_')}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f" Confusion matrix saved to: {save_path}")

    plt.close()

if __name__ == "__main__":
    evaluate_model("models/random_forest.pkl", "Random Forest")
    evaluate_model("models/svm.pkl", "SVM")
