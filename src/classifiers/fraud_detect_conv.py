import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from datetime import datetime
import sys

sys.path.append("../")

from tools.EnergyMeter.energy_meter import EnergyMeter


# RobustLabelEncoder and extract_time_features remain the same
class RobustLabelEncoder:
    def __init__(self):
        self.mapping = {}
        self.counter = 0

    def fit(self, data):
        unique_values = pd.Series(data).unique()
        for value in unique_values:
            self.mapping[value] = self.counter
            self.counter += 1
        return self

    def transform(self, data):
        return pd.Series(data).map(lambda x: self.mapping.get(x, self.counter))

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)


def calculate_custom_metrics(
    y_true,
    y_pred,
    y_prob=None,
    cost_weights={"fp": 1, "fn": 5},
):
    """
    Calculate custom evaluation metrics for fraud detection

    Parameters:
    - y_true: True labels
    - y_pred: Predicted labels
    - y_prob: Prediction probabilities (optional)
    - cost_weights: Dictionary with weights for false positives and negatives

    Returns:
    - Dictionary of custom metrics
    """
    # Calculate confusion matrix elements
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))

    # Calculate basic metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Calculate weighted metrics
    weighted_recall = (cost_weights["fn"] * recall + recall) / (cost_weights["fn"] + 1)
    weighted_precision = (cost_weights["fp"] * precision + precision) / (
        cost_weights["fp"] + 1
    )

    # Calculate weighted F1
    weighted_f1 = (
        2
        * (weighted_precision * weighted_recall)
        / (weighted_precision + weighted_recall)
        if (weighted_precision + weighted_recall) > 0
        else 0
    )

    # Calculate cost-sensitive error
    total_cost = (cost_weights["fp"] * fp + cost_weights["fn"] * fn) / len(y_true)

    # Calculate AURPC if probabilities are available
    aurpc = None
    if y_prob is not None:
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        # Fix: Sort by recall for proper integration
        sorted_idx = np.argsort(recall_curve)
        recall_curve = recall_curve[sorted_idx]
        precision_curve = precision_curve[sorted_idx]
        # Calculate area using proper trapezoidal rule
        aurpc = np.trapz(precision_curve, x=recall_curve)
        # Ensure positive value
        aurpc = abs(aurpc)

    return {
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "cost_sensitive_error": total_cost,
        "aurpc": aurpc,
    }


def extract_time_features(df):
    """Extract features from transaction timestamp and DOB"""
    df["trans_datetime"] = pd.to_datetime(
        df["trans_date_trans_time"], format="%Y-%m-%d %H:%M:%S"
    )

    df["trans_hour"] = df["trans_datetime"].dt.hour.astype(float)
    df["trans_day"] = df["trans_datetime"].dt.day.astype(float)
    df["trans_month"] = df["trans_datetime"].dt.month.astype(float)
    df["trans_day_of_week"] = df["trans_datetime"].dt.dayofweek.astype(float)
    df["is_weekend"] = df["trans_datetime"].dt.dayofweek.isin([5, 6]).astype(float)

    df["dob_datetime"] = pd.to_datetime(df["dob"], format="%Y-%m-%d")
    df["age"] = (df["trans_datetime"] - df["dob_datetime"]).dt.days / 365.25
    df["age"] = df["age"].astype(float)
    df["age_decade"] = (df["age"] // 10 * 10).astype(float)

    df = df.drop(
        ["trans_datetime", "dob_datetime", "trans_date_trans_time", "dob"], axis=1
    )

    return df


def train_fraud_detection(
    train_path,
    test_path=None,
    fraud_ratio=0.3,
    sample_size=100000,
    cost_weights={"fp": 1, "fn": 5},
    print_metrics=False,
):
    """Train fraud detection models with energy metering"""
    # Initialize energy meter
    meter = EnergyMeter(
        disk_avg_speed=1600 * 1e6,
        disk_active_power=6,
        disk_idle_power=1.42,
        label="Power usage LLM query",
        include_idle=False,
    )

    energy_results = {}

    # Start measuring energy consumption
    meter.begin()

    print("Loading training data...")
    train_data = pd.read_csv(train_path)

    drop_cols = [
        "Unnamed: 0",
        "cc_num",
        "first",
        "last",
        "street",
        "city",
        "state",
        "zip",
        "trans_num",
    ]
    train_data = train_data.drop(
        columns=[col for col in drop_cols if col in train_data.columns]
    )

    print("Extracting time features...")
    train_data = extract_time_features(train_data)

    fraud_data = train_data[train_data["is_fraud"] == 1]
    non_fraud_data = train_data[train_data["is_fraud"] == 0]

    print(
        f"Original dataset - Fraud cases: {len(fraud_data)}, Non-fraud cases: {len(non_fraud_data)}"
    )

    n_fraud = min(len(fraud_data), int(sample_size * fraud_ratio))
    n_non_fraud = min(len(non_fraud_data), int(sample_size * (1 - fraud_ratio)))

    fraud_sampled = fraud_data.sample(n=n_fraud, random_state=42)
    non_fraud_sampled = non_fraud_data.sample(n=n_non_fraud, random_state=42)

    train_data = pd.concat([fraud_sampled, non_fraud_sampled])
    train_data = shuffle(train_data, random_state=42)

    print(
        f"Training dataset - Fraud cases: {len(fraud_sampled)}, Non-fraud cases: {len(non_fraud_sampled)}"
    )

    X = train_data.drop("is_fraud", axis=1)
    y = train_data["is_fraud"]

    if test_path:
        print("Loading test data...")
        test_data = pd.read_csv(test_path)
        test_data = test_data.drop(
            columns=[col for col in drop_cols if col in test_data.columns]
        )
        test_data = extract_time_features(test_data)
        X_test = test_data.drop("is_fraud", axis=1)
        y_test = test_data["is_fraud"]
    else:
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X, y = X_train, y_train

    print("Preprocessing data...")

    categorical_cols = ["merchant", "category", "gender", "job"]
    encoders = {}
    for col in categorical_cols:
        encoders[col] = RobustLabelEncoder()
        X[col] = encoders[col].fit_transform(X[col])
        X_test[col] = encoders[col].transform(X_test[col])

    numerical_cols = [
        "amt",
        "lat",
        "long",
        "city_pop",
        "unix_time",
        "merch_lat",
        "merch_long",
        "trans_hour",
        "trans_day",
        "trans_month",
        "trans_day_of_week",
        "is_weekend",
        "age",
        "age_decade",
    ]
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    X = X.to_numpy()
    y = y.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    meter.end()
    energy_results["preprocess"] = meter.get_total_jules_per_component()
    meter = EnergyMeter(
        disk_avg_speed=1600 * 1e6,
        disk_active_power=6,
        disk_idle_power=1.42,
        label="Power usage LLM query",
        include_idle=False,
    )
    meter.begin()
    # Train SVM model
    print("Training SVM model on CPU...")
    from sklearn.svm import SVC  # Import sklearn's SVC

    # Replace the SVC model with LinearSVC
    base_svm = LinearSVC(
        C=10.0,
        loss="hinge",
        dual=True,
        max_iter=1000,
        class_weight={0: 1, 1: (1 - fraud_ratio) / fraud_ratio},
        random_state=42,
    )

    # Wrap with CalibratedClassifierCV to get probability estimates
    svm_model = CalibratedClassifierCV(base_svm, cv=5)
    svm_model.fit(X, y)

    meter.end()
    energy_results["SVC_train"] = meter.get_total_jules_per_component()
    meter = EnergyMeter(
        disk_avg_speed=1600 * 1e6,
        disk_active_power=6,
        disk_idle_power=1.42,
        label="Power usage LLM query",
        include_idle=False,
    )
    meter.begin()
    # Train Random Forest model
    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        max_features="sqrt",
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        # Adding class weight to match the SVM approach
        class_weight={0: 1, 1: (1 - fraud_ratio) / fraud_ratio},
    )
    rf_model.fit(X, y)

    meter.end()
    energy_results["RF_train"] = meter.get_total_jules_per_component()

    # Modify the evaluation section in the original code:
    print("Evaluating models...")
    models = {"SVM": svm_model, "Random Forest": rf_model}

    for name, model in models.items():
        meter = EnergyMeter(
            disk_avg_speed=1600 * 1e6,
            disk_active_power=6,
            disk_idle_power=1.42,
            label="Power usage LLM query",
            include_idle=False,
        )
        meter.begin()
        print(f"\n{name} Results:")
        y_pred = model.predict(X_test)

        # Get prediction probabilities for AURPC if available
        try:
            y_prob = model.predict_proba(X_test)[:, 1]
        except (AttributeError, NotImplementedError):
            y_prob = None

        if print_metrics:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred))

            # Calculate and print custom metrics
            custom_metrics = calculate_custom_metrics(
                y_test, y_pred, y_prob, cost_weights
            )
            print("\nCustom Metrics:")
            print(f"Weighted Recall: {custom_metrics['weighted_recall']:.4f}")
            print(f"Weighted F1: {custom_metrics['weighted_f1']:.4f}")
            print(f"Cost-sensitive Error: {custom_metrics['cost_sensitive_error']:.4f}")
            print(f"AURPC: {custom_metrics['aurpc']:.4f}")

            plt.figure(figsize=(8, 6), dpi=600)
            cm = pd.crosstab(
                y_test, y_pred, rownames=["Actual"], colnames=["Predicted"]
            )
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title(f"Confusion Matrix - {name}")
            plt.show()

        meter.end()
        energy_stats = meter.get_total_jules_per_component()
        energy_results[name] = energy_stats

    # Assuming your dictionary is named 'data'
    for model_dict in energy_results.values():
        model_dict.pop("dram")
        model_dict.pop("disk")
    from pprint import pprint

    pprint(energy_results)

    return {"svm": svm_model, "rf": rf_model}, encoders, scaler, energy_results


# Example usage with custom weights:
models, encoders, scaler, energy = train_fraud_detection(
    train_path="/home/hessel/code/fraudTrain.csv",
    test_path="/home/hessel/code/fraudTest.csv",
    fraud_ratio=0.62,
    sample_size=100000,
    cost_weights={"fp": 1, "fn": 10},  # Higher penalty for false negatives
)
