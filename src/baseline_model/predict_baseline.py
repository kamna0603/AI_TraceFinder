import os
import cv2
import numpy as np
import pandas as pd
import joblib
from scipy.stats import skew, kurtosis, entropy
from skimage.filters import sobel

CSV_REF = "processed_data/metadata_features.csv"
MODELS_DIR = "models"

def load_and_preprocess(img_path, size=(512, 512)):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"❌ Could not load image: {img_path}")
    img = img.astype(np.float32) / 255.0
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

def compute_metadata_features(img, file_path):
    h, w = img.shape
    aspect_ratio = w / h
    file_size_kb = os.path.getsize(file_path) / 1024

    pixels = img.flatten()
    mean_intensity = np.mean(pixels)
    std_intensity = np.std(pixels)
    skewness = skew(pixels)
    kurt = kurtosis(pixels)
    ent = entropy(np.histogram(pixels, bins=256, range=(0,1))[0] + 1e-6)
    edges = sobel(img)
    edge_density = np.mean(edges > 0.1)

    return {
        "width": w,
        "height": h,
        "aspect_ratio": aspect_ratio,
        "file_size_kb": file_size_kb,
        "mean_intensity": mean_intensity,
        "std_intensity": std_intensity,
        "skewness": skewness,
        "kurtosis": kurt,
        "entropy": ent,
        "edge_density": edge_density
    }

def predict_scanner(img_path, model_choice="rf"):
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    if model_choice == "rf":
        model = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    else:
        model = joblib.load(os.path.join(MODELS_DIR, "svm.pkl"))

    img = load_and_preprocess(img_path)
    features = compute_metadata_features(img, img_path)

    ref = pd.read_csv(CSV_REF)
    feature_cols = [c for c in ref.columns if c not in ["file_name","main_class","resolution","class_label"]]
    df = pd.DataFrame([features])[feature_cols]
    X_scaled = scaler.transform(df)

    pred = model.predict(X_scaled)[0]
    prob = dict(zip(model.classes_, model.predict_proba(X_scaled)[0]))

    return pred, prob

if __name__ == "__main__":
    test_image = r"D:\AI_TraceFinder\Data\Official\Official-20250910T085215Z-1-001\Official\Canon120-1\150\s1_4.tif"
    pred, prob = predict_scanner(test_image, "rf")
    print("Predicted Scanner:", pred)
    print("Class Probabilities:", prob)
   

