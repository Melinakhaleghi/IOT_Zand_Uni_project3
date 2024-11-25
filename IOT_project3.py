# Install Kaggle and import libraries
pip install kaggle
import os
import zipfile
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Define dataset and download
dataset_name = "andersy005/cicids-2017"
os.system(f"kaggle datasets download -d {dataset_name}")

# Extract dataset files
with zipfile.ZipFile("cicids-2017.zip", 'r') as zip_ref:
    zip_ref.extractall("cicids_2017_dataset")

# Load dataset
df = pd.read_csv("cicids_2017_dataset/some_file.csv")

# Combine files into a single DataFrame
all_data = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)
all_data = all_data.drop(columns=['irrelevant_column_1', 'irrelevant_column_2'], errors='ignore')

# Normalize data
scaler = StandardScaler()
features = scaler.fit_transform(all_data.drop(columns=['Label']))
labels = all_data['Label']

# Define LSTM model
model = Sequential([
    LSTM(128, input_shape=(100, features.shape[1]), return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Create sequences
def create_sequences(data, labels, seq_length=100):
    return (np.array([data[i:i + seq_length] for i in range(len(data) - seq_length)]), 
                     np.array(labels[seq_length:]))

X, y = create_sequences(features, labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate model
y_pred = model.predict(X_test) > 0.5
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))

# Identify misclassifications
df_errors = pd.DataFrame({
    "Actual Label": y_test,
    "Predicted Label": y_pred.flatten()
})

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Benign", "Attack"]).plot()

# Feature scatter plot
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap="coolwarm", alpha=0.6, label="Actual")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred.flatten(), cmap="viridis", alpha=0.3, label="Predicted")
plt.legend()
plt.show()

