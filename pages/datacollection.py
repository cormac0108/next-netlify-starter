import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained emotion detection model
model = load_model('emotion_detection_model.h5')

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Load the Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create empty lists to store the collected data
images = []
labels = []

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = np.expand_dims(face_roi, axis=-1)
        face_roi = face_roi / 255.0

        # Perform emotion detection inference
        predictions = model.predict(face_roi)
        predicted_index = np.argmax(predictions)
        predicted_emotion = emotion_labels[predicted_index]

        # Draw bounding box and label for the detected face
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Collect the face ROI and corresponding label
        images.append(face_roi)
        labels.append(predicted_index)

    # Display the frame
    cv2.imshow('Emotion Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()

# Convert the collected data to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Save the collected data
np.save('data_images.npy', images)
np.save('data_labels.npy', labels)
