import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import config

def train_model():
    """
    Trains a Random Forest model using the mock training data.
    Saves the model to the path specified in config.py.
    """
    data_path = os.path.join(config.BASE_DIR, "training_data.csv")
    if not os.path.exists(data_path):
        print("Training data not found. Please run data_pipeline.py first.")
        return None
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    X = df[config.FEATURES]
    y = df[config.TARGET]
    
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Save model
    print(f"Saving model to {config.MODEL_PATH}...")
    joblib.dump(model, config.MODEL_PATH)
    print("Model training complete!")
    
    return {
        "accuracy": float(acc),
        "roc_auc": float(roc_auc)
    }

if __name__ == "__main__":
    train_model()
