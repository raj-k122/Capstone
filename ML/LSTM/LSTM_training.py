import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from create_training_data import load_pose_data_and_create_labels, preprocess_sequences

#Load data
pose_data_directory = "processed_pose_data"
pose_sequences, labels = load_pose_data_and_create_labels(pose_data_directory, augment=True, augmentation_healthy_factor=50,augmentation_hemi_factor=250, noise_level=0.1)

padded_pose_sequences = preprocess_sequences(pose_sequences)

X_train, X_temp, y_train, y_temp = train_test_split(padded_pose_sequences, labels, test_size=0.3, random_state=42, stratify=labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(128, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # Binary classification (healthy/hemiplegic)
    ])
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Get the input shape (time steps, features)
input_shape = (X_train.shape[1], X_train.shape[2])
lstm_model = create_lstm_model(input_shape)
lstm_model.summary()

epochs = 50  
batch_size = 16

early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model
history = lstm_model.fit(
    X_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)
lstm_model.save("hemiplegia_classifier_model.h5") #save model.
# Evaluate the model on the test set
loss, accuracy = lstm_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")