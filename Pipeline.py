import Audio_Plotting
import Audio_Extraction
import Feature_Extraction
import pandas as pd
import os

def run_full_audio_pipeline_to_features(file_path: str) :
    features = Feature_Extraction.extract_combined_features(file_path)
    return pd.DataFrame([features])

def run_batch_audio_pipeline(folder_path: str):
    all_feature_rows = []

    for fname in os.listdir(folder_path):
        full_path = os.path.join(folder_path, fname)
        print(f"Processing: {fname}")
        features_df = run_full_audio_pipeline_to_features(full_path)
        all_feature_rows.append(features_df)

    if all_feature_rows:
        big_df = pd.concat(all_feature_rows, ignore_index=True)
    else:
        big_df = pd.DataFrame()

    return big_df

real_folder_path = "C:\\Users\\habel\\OneDrive\\Desktop\\UCL\\SDE Anomaly Detection\\Anthem_Segment_Audio"
fake_folder_path = "C:\\Users\\habel\\OneDrive\\Desktop\\UCL\\SDE Anomaly Detection\\fake_clips\\Chap_All"

real_features_df = run_batch_audio_pipeline(real_folder_path)
fake_features_df = run_batch_audio_pipeline(fake_folder_path)

import Classifier_Algorithim

clf, y_prob, shap_values = Classifier_Algorithim.train_audio_classifier(real_features_df, fake_features_df)
