import cv2
import mediapipe as mp
import numpy as np
import os

def extract_and_normalize_pose(video_path, output_dir="pose_data"):
    """
    Extracts pose data from a video, normalizes it, and saves it to a .npy file.

    Args:
        video_path (str): Path to the input video file (mp4).
        output_dir (str, optional): Directory to save the pose data. Defaults to "pose_data".
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    pose_data = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            frame_pose_data = []
            for landmark in landmarks:
                frame_pose_data.extend([landmark.x, landmark.y])  # 2D coordinates
            pose_data.append(frame_pose_data)
        else:
            # Handle missing pose data (fill with zeros)
            frame_pose_data = [0.0] * (len(mp_pose.PoseLandmark) * 2)
            pose_data.append(frame_pose_data)

    cap.release()
    cv2.destroyAllWindows()

    if not pose_data:
        print(f"Warning: No pose data extracted from {video_path}")
        return

    normalized_pose_data = normalize_pose_data(np.array(pose_data))

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a filename based on the video filename
    video_filename_without_ext = os.path.splitext(os.path.basename(video_path))[0]
    output_filename = os.path.join(output_dir, f"{video_filename_without_ext}_pose.npy")

    # Save the normalized pose data
    np.save(output_filename, normalized_pose_data)
    print(f"Normalized pose data saved to: {output_filename}")

def normalize_pose_data(pose_data):
    """
    Normalizes pose data to the range [-1, 1].

    Args:
        pose_data (numpy.ndarray): Pose data.

    Returns:
        numpy.ndarray: Normalized pose data.
    """
    if pose_data is None or len(pose_data) == 0:
        return None

    max_val = np.max(np.abs(pose_data))
    if max_val == 0:
        return pose_data #prevent divide by zero errors.

    return pose_data / max_val

def process_video_folder(video_folder, output_directory="pose_data"):
    """
    Processes all video files in a given folder and saves their pose data.

    Args:
        video_folder (str): Path to the folder containing video files.
        output_directory (str, optional): Directory to save the pose data. Defaults to "pose_data".
    """
    for filename in os.listdir(video_folder):
        if filename.endswith(".mp4"):  # You can add other video extensions if needed
            video_path = os.path.join(video_folder, filename)
            extract_and_normalize_pose(video_path, output_directory)

if __name__ == "__main__":
    healthy_folder = "/Users/rajkulkarni/Documents/UTS 2025/Capstone/MediaPipe/videos/original footage/version 2 dataset (healthy)"  # Replace with your healthy video folder path
    hemiplegic_folder = "/Users/rajkulkarni/Documents/UTS 2025/Capstone/MediaPipe/videos/patients/patient #6/front"  # Replace with your hemiplegic video folder path
    testing_folder = "/Users/rajkulkarni/Documents/UTS 2025/Capstone/MediaPipe/videos/original footage/testing/patient #7/"
    output_pose_dir = "processed_pose_data/testing/"

    # Create the main output directory if it doesn't exist
    os.makedirs(output_pose_dir, exist_ok=True)

    # healthy_output_dir = os.path.join(output_pose_dir, "healthy")
    # hemiplegic_output_dir = os.path.join(output_pose_dir, "hemiplegic/front")


    process_video_folder(testing_folder, output_pose_dir)

    print("Pose data extraction and normalization complete. Check the 'processed_pose_data' folder.")