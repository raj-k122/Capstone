import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 1. Load your trained model
model_path = 'training_runs/model_run_6/model_run_6.h5'
loaded_model = keras.models.load_model(model_path)

# 2. Load your new pose sequence NumPy array
new_pose_sequence = np.load('processed_pose_data/testing/front_p9_testing_pose.npy') # Replace with the actual path

# 3. Get the max_length from the model summary
max_length = 977

# 4. Pad the new sequence
padded_new_sequence = pad_sequences([new_pose_sequence], maxlen=max_length, padding='post', dtype='float32')

# 5. Make the prediction (no explicit reshape needed as pad_sequences handles the batch dimension)
predictions = loaded_model.predict(padded_new_sequence)

# 6. Process the predictions
print("Predictions shape:", predictions.shape)
print("Prediction (probability of being abnormal:", predictions[0, 0])

probability_hemiplegic = predictions[0, 0]
threshold = 0.7

if probability_hemiplegic > threshold:
    print("Prediction: Abnormal")
else:
    print("Prediction: Healthy")