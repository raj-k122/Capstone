import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

def load_pose_data_and_create_labels(pose_data_dir, augment=False, augmentation_healthy_factor=2,augmentation_hemi_factor=2, noise_level=0.01):
    """
    Loads normalized pose data, creates labels, and optionally augments the data.

    Args:
        pose_data_dir (str): Path to the main directory.
        augment (bool): Whether to perform data augmentation.
        augmentation_factor (int): Number of augmented samples per original sample.
        noise_level (float): Standard deviation of the noise added.

    Returns:
        tuple: Padded sequences and labels.
    """
    healthy_dir = os.path.join(pose_data_dir, "healthy/forward")
    hemiplegic_dir = os.path.join(pose_data_dir, "hemiplegic/front")

    pose_sequences = []
    labels = []

    def add_noise(pose_sequence, noise_level=noise_level):
        """Adds random noise to the pose coordinates."""
        noisy_sequence = pose_sequence + np.random.normal(0, noise_level, pose_sequence.shape)
        return noisy_sequence
    
    def process_healthy_directory(directory, label):
        for filename in os.listdir(directory):
            if filename.endswith("_pose.npy"):
                file_path = os.path.join(directory, filename)
                pose_data = np.load(file_path)
                pose_sequences.append(pose_data)
                labels.append(label)

                if augment:
                    for _ in range(augmentation_healthy_factor - 1): # -1 because we include the original.
                        augmented_data = add_noise(pose_data.copy()) # Use a copy to avoid altering the original
                        pose_sequences.append(augmented_data)
                        labels.append(label)

    def process_hemiplegic_directory(directory, label):
        for filename in os.listdir(directory):
            if filename.endswith("_pose.npy"):
                file_path = os.path.join(directory, filename)
                pose_data = np.load(file_path)
                pose_sequences.append(pose_data)
                labels.append(label)

                if augment:
                    for _ in range(augmentation_hemi_factor - 1): # -1 because we include the original.
                        augmented_data = add_noise(pose_data.copy()) # Use a copy to avoid altering the original
                        pose_sequences.append(augmented_data)
                        labels.append(label)

    process_healthy_directory(healthy_dir, 0) # 0 for healthy
    process_hemiplegic_directory(hemiplegic_dir, 1) # 1 for hemiplegic

    return pose_sequences, np.array(labels)

def preprocess_sequences(pose_sequences):
    """Pads sequences to the maximum length."""
    max_length = max(len(seq) for seq in pose_sequences)
    padded_sequences = pad_sequences(pose_sequences, maxlen=max_length, padding='post', dtype='float32')
    return padded_sequences

if __name__ == "__main__":
    pose_data_directory = "processed_pose_data"
    augment_data = True # Enable or disable augmentation
    augmentation_factor = 250 # Number of augmented samples per original (including original)
    noise_strength = 0.01 # Adjust noise level as needed

    pose_sequences, labels = load_pose_data_and_create_labels(pose_data_directory, augment=augment_data, augmentation_healthy_factor=augmentation_factor, augmentation_hemi_factor=augmentation_factor, noise_level=noise_strength)

    if not pose_sequences:
        print("No pose data found.")
    else:
        padded_pose_sequences = preprocess_sequences(pose_sequences)

        X_train, X_temp, y_train, y_temp = train_test_split(padded_pose_sequences, labels, test_size=0.3, random_state=42, stratify=labels)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

        print("Shape of X_train:", X_train.shape)
        print("Shape of y_train:", y_train.shape)
        print("Shape of X_val:", X_val.shape)
        print("Shape of y_val:", y_val.shape)
        print("Shape of X_test:", X_test.shape)
        print("Shape of y_test:", y_test.shape)



