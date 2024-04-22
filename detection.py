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
# cap = cv2.VideoCapture('http://192.168.0.101:8080/video')
cap.set(cv2.CAP_PROP_FPS, 60)  # Change 30 to your desired frame rate
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

    # Display the activity label on the frame
    cv2.putText(
        frame, activity, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    # Display the frame
    cv2.imshow('Human Activity Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(classes[predicted_class])
    # print(prediction)
# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
