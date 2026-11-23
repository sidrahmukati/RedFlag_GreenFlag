import os
import joblib
import pandas as pd
from schema import PILLAR_COLUMNS
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier


def train_and_save_model():
    # 1. Load training_data.csv and separate features (X) from target (y)
    data_path = os.path.join("data", "training_data.csv")
    df = pd.read_csv(data_path)
    X = df[PILLAR_COLUMNS]
    y = df["target"]

    # 2. Initialize XGBoost with your chosen hyperparameters
    model = XGBClassifier(
        random_state=42,
        n_estimators=100,
        max_depth=3,
        learning_rate=0.08,
        eval_metric="logloss",
    )

    # 3. Evaluate with 5-Fold Stratified Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X, y, cv=skf, scoring=["accuracy", "roc_auc"], return_train_score=True
    )

    print(f"5-Fold Train Accuracy : {cv_results['train_accuracy'].mean():.2%}")
    print(f"5-Fold Test Accuracy  : {cv_results['test_accuracy'].mean():.2%}")
    print(f"5-Fold Test ROC-AUC   : {cv_results['test_roc_auc'].mean():.4f}")

    # 4. Fit on 100% of the data and save to disk
    model.fit(X, y)
    joblib.dump(model, "flag_model.joblib")
    print("\nSaved trained model to flag_model.joblib!")


if __name__ == "__main__":
    train_and_save_model()