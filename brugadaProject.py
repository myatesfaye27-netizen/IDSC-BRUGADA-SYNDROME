# Import libraries
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load metadata
metadata = pd.read_csv("metadata.csv")

# Convert to binary labels
metadata['brugada'] = metadata['brugada'].apply(lambda x: 1 if x > 0 else 0)

print("First few rows of metadata:")
print(metadata.head())

print("\nDataset statistics")
print("Total subjects:", len(metadata))
print("Brugada patients:", (metadata['brugada'] == 1).sum())
print("Healthy subjects:", (metadata['brugada'] == 0).sum())


# Feature extraction
def extract_ecg_features(signals):

    features = []

    for i in range(signals.shape[1]):
        lead_signal = signals[:, i]
        mean_val = np.mean(lead_signal)
        std_val = np.std(lead_signal)
        max_val = np.max(lead_signal)
        min_val = np.min(lead_signal)

        features.extend([mean_val, std_val, max_val, min_val])

    return features

# Read ECG files
data_features = []
labels = []

base_path = "C:/Users/user/PyCharmMiscProject/brugada-huca/files"
print("\nReading ECG signals...")

for index, row in metadata.iterrows():
    patient_id = str(row['patient_id'])
    label = row['brugada']

    try:
        record_path = os.path.join(base_path, patient_id, patient_id)
        record = wfdb.rdrecord(record_path)
        signals = record.p_signal
        features = extract_ecg_features(signals)
        data_features.append(features)
        labels.append(label)

    except Exception as e:
        print("Error reading patient:", patient_id)
        print(e)

X = np.array(data_features)
y = np.array(labels)

print("\nFeature matrix shape:", X.shape)

if len(X) == 0:
    raise ValueError("No ECG data loaded. Check dataset path.")

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Normalize
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Train model
print("\nTraining Machine Learning Model...")

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)


# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# Prediction function
def predict_patient(patient_id):

    try:
        record_path = os.path.join(base_path, patient_id, patient_id)
        record = wfdb.rdrecord(record_path)
        signals = record.p_signal


        # ECG Visualization
        plt.figure(figsize=(10,4))
        plt.plot(signals[:1000,6])   # Lead V1
        plt.title(f"ECG Signal (Lead V1) - Patient {patient_id}")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.show()


        features = extract_ecg_features(signals)
        features = scaler.transform([features])
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]

        if prediction == 1:
            result = "Brugada detected"
            confidence = probability[1]
        else:
            result = "Normal ECG"
            confidence = probability[0]

        print("\nPrediction for patient", patient_id)
        print("Result:", result)
        print("Confidence:", round(confidence * 100, 2), "%")

    except Exception as e:
        print("Error predicting patient:", patient_id)
        print(e)


# Prediction sample
predict_patient("188981")