import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from create_training_data import load_pose_data_and_create_labels, preprocess_sequences
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

pose_data_directory = "processed_pose_data"
pose_sequences, labels = load_pose_data_and_create_labels(pose_data_directory, augment=True, augmentation_healthy_factor=50,augmentation_hemi_factor=250, noise_level=0.1)

padded_pose_sequences = preprocess_sequences(pose_sequences)

X_train, X_temp, y_train, y_temp = train_test_split(padded_pose_sequences, labels, test_size=0.3, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

model = load_model("training_runs/model_run_1/model_run_1.h5")
# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")
# Predict class probabilities
y_prob = model.predict(X_test)

# Convert to class predictions
y_pred = np.argmax(y_prob, axis=1) if y_prob.shape[1] > 1 else (y_prob > 0.5).astype(int).flatten()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Healthy", "Abnormal"]))

# Print and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["Healthy", "Abnormal"],
            yticklabels=["Healthy", "Abnormal"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()