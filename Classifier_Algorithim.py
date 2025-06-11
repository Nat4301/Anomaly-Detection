
import os
import json
from pydub import AudioSegment
from tqdm import tqdm
from scipy.stats import gaussian_kde

import Feature_Extraction

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import shap
import matplotlib.pyplot as plt
import seaborn as sns

def build_kde_estimators(df_real, df_fake):
    features = [col for col in df_real.columns if col not in ["filename"]]
    kde_real = {}
    kde_fake = {}
    for feat in features:
        kde_real[feat] = gaussian_kde(df_real[feat])
        kde_fake[feat] = gaussian_kde(df_fake[feat])
    return kde_real, kde_fake, features


def classify_with_kde(file_path, kde_real, kde_fake, features):
    try:
        features_vals = Feature_Extraction.extract_combined_features(file_path)

        weighted_votes = {'REAL': 0.0, 'FAKE': 0.0}
        feature_results = {}

        for feat in features:
            val = features_vals.get(feat, None)
            if val is None:
                continue  # skip missing features

            p_real = kde_real[feat].evaluate(val)[0]
            p_fake = kde_fake[feat].evaluate(val)[0]

            total_p = p_real + p_fake
            if total_p == 0:
                prob_real = prob_fake = 0.5
            else:
                prob_real = p_real / total_p
                prob_fake = p_fake / total_p

            if prob_real > prob_fake:
                decision = 'REAL'
                confidence = prob_real - prob_fake
            else:
                decision = 'FAKE'
                confidence = prob_fake - prob_real

            weighted_votes[decision] += confidence

            feature_results[feat] = {
                'value': val,
                'prob_real': prob_real,
                'prob_fake': prob_fake,
                'decision': decision,
                'confidence': confidence
            }

        # Final decision and confidence
        total_confidence = weighted_votes['REAL'] + weighted_votes['FAKE']
        if total_confidence == 0:
            final_pred = 'UNDECIDED'
            final_conf = 0.0
        elif weighted_votes['REAL'] > weighted_votes['FAKE']:
            final_pred = 'REAL'
            final_conf = weighted_votes['REAL'] / total_confidence
        elif weighted_votes['FAKE'] > weighted_votes['REAL']:
            final_pred = 'FAKE'
            final_conf = weighted_votes['FAKE'] / total_confidence
        else:
            final_pred = 'UNDECIDED'
            final_conf = 0.0

        return {
            'prediction': final_pred,
            'confidence': f"{final_conf * 100:.1f}%",
            'weighted_votes': {k: float(v) for k, v in weighted_votes.items()},
            'features': feature_results
        }

    except Exception as e:
        return {'error': str(e)}


def train_audio_classifier(real_df: pd.DataFrame, fake_df: pd.DataFrame):
    # Label the data
    real_df = real_df.copy()
    fake_df = fake_df.copy()
    real_df["label"] = 1  # 1 for real
    fake_df["label"] = 0  # 0 for fake

    # Combine datasets
    full_df = pd.concat([real_df, fake_df], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    # Separate features and label
    X = full_df.drop(columns=["filename", "label"], errors='ignore')
    y = full_df["label"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose classifier - RandomForest often performs well on tabular data
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predictions and confidence scores
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]  # Probability of class '1' (real)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\nModel Accuracy: {:.4f}".format(accuracy))
    print("\nClassification Report:\n", report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

    # Feature importances
    importances = clf.feature_importances_
    indices = np.argsort(importances)
    feature_names = np.array(X.columns)[indices]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=feature_names)
    plt.title("Feature Importances Order")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

    # SHAP analysis
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test)

    print("\nGenerating SHAP summary plot...")
    shap.summary_plot(shap_values[1], X_test, plot_type="bar")

    return clf, y_prob, shap_values


