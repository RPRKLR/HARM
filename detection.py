import cv2
import numpy as np
from keras.models import load_model

# Load your pre-trained Keras model
model = load_model(
    '/home/peter/dp/Human_Activity_Video_Recognition/data/output/models/DVORAK_CUSTOM/human_activity_recognition_model_4_classes_adam_0_005__epochs_200__batch_size_128__early_stopping_monitor_val_loss_mode_min_patience_15_2024-03-26_14:21:44_88percent_acc.keras'
)

# Define the classes corresponding to human activities
classes = [
    'high_jumps',
    'jumping_jacks',
    'lunges',
    'squat',
]  # Add your activity classes here

# Define the mapping of classes to video files
class_to_video = {
    'high_jumps': 'data/ai_generated_videos_for_popularization/high_jumps.mp4',
    'jumping_jacks': 'data/ai_generated_videos_for_popularization/jumping_jacks.mp4',
    'lunges': 'data/ai_generated_videos_for_popularization/lunges.mp4',
    'squat': 'data/ai_generated_videos_for_popularization/squats.mp4',
}


def preprocess_frame(frame):
    # Resize the frame to match the input shape of your model
    resized_frame = cv2.resize(frame, (64, 64))
    # Expand dimensions to match the expected batch size of 20
    expanded_frame = np.expand_dims(resized_frame, axis=0)
    # Repeat the frame along the first axis to match the batch size of 20
    processed_frame = np.repeat(expanded_frame, 20, axis=0)
    return processed_frame


# Access webcam feed
cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

# Set the width and height of the frame
frame_width = 1080
frame_height = 1280
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Set the frame rate
cap.set(cv2.CAP_PROP_FPS, 60)  # Change 30 to your desired frame rate

# Initialize variables to store the current class and video file
current_class = None
current_video_path = None
current_video_cap = None

# Define window sizes
webcam_window_size = (frame_width // 2, frame_height // 2)
video_window_size = (frame_width // 2, frame_height // 2)

# Set the factor by which to increase the video playback speed
# Increase this value to skip more frames and play the video faster
skip_frames_factor = 3  # Adjust this value according to your preference

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Perform inference using your Keras model
    prediction = model.predict(np.expand_dims(processed_frame, axis=0))
    predicted_class = np.argmax(prediction)

    # Get the class label
    activity = classes[predicted_class]

    # If the detected class is different from the current class,
    # open the corresponding video file
    if activity != current_class:
        current_class = activity
        if current_video_cap is not None:
            current_video_cap.release()
        current_video_path = class_to_video.get(current_class, None)
        if current_video_path is not None:
            current_video_cap = cv2.VideoCapture(current_video_path)
            if not current_video_cap.isOpened():
                print('Error: Unable to open video file:', current_video_path)
                current_video_cap = None

    # Display the activity label on the frame
    cv2.putText(
        frame, activity, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    # Display the frame from webcam
    cv2.imshow(
        'Human Activity Detection - Webcam',
        cv2.resize(frame, webcam_window_size),
    )

    # If a video file is open for the current class, display the frame from the video
    if current_video_cap is not None:
        # Read and skip frames to increase playback speed
        for _ in range(skip_frames_factor):
            current_video_cap.grab()
        ret, video_frame = current_video_cap.read()
        if not ret:
            current_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _, video_frame = current_video_cap.read()
        cv2.imshow('Activity Video', cv2.resize(video_frame, video_window_size))

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(activity)

# Release the webcam, video capture, and close the windows
cap.release()
if current_video_cap is not None:
    current_video_cap.release()
cv2.destroyAllWindows()
