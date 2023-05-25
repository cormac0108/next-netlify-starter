import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model
model = load_model('emotion_detection_model.h5')

# Load the test dataset
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')

# Preprocess the test data
num_classes = len(np.unique(test_labels))
test_labels = to_categorical(test_labels, num_classes)
test_images = test_images.reshape(test_images.shape[0], 48, 48, 1)

# Perform inference on the test dataset
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Evaluate the model
accuracy = accuracy_score(np.argmax(test_labels, axis=1), predicted_labels)
report = classification_report(np.argmax(test_labels, axis=1), predicted_labels)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)
