import sys
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import json

# Load the trained model and class labels
model = tf.keras.models.load_model('dog_breed_classifier.keras')
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)
breed_names = {v: k for k, v in class_labels.items()}

img_size = (224, 224)

def predict_breed(image_path):
    img = load_img(image_path, target_size=img_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return breed_names[predicted_class]

class DogBreedApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dog Breed Classifier")
        self.setGeometry(100, 100, 600, 600)

        # Set up layout
        layout = QVBoxLayout()

        # Display area for images or camera feed, size will be adjusted based on aspect ratio
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

        # Prediction label
        self.prediction_label = QLabel("Predicted Breed: ", self)
        self.prediction_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.prediction_label)

        # Buttons
        self.capture_button = QPushButton("Capture Picture", self)
        self.capture_button.clicked.connect(self.capture_image)
        layout.addWidget(self.capture_button)

        self.upload_button = QPushButton("Upload Picture", self)
        self.upload_button.clicked.connect(self.upload_image)
        layout.addWidget(self.upload_button)

        # Initialize camera feed
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.prediction_label.setText("Error: Could not open camera.")
            self.capture_button.setEnabled(False)
        else:
            # Timer to update the camera feed in the label
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_camera_feed)
            self.timer.start(30)  # Update every 30 ms for a smooth video feed

        # Set layout
        self.setLayout(layout)

    def update_camera_feed(self):
        ret, frame = self.cap.read()
        if ret:
            # Determine the aspect ratio of the frame
            h, w, _ = frame.shape
            aspect_ratio = w / h
            
            # Adjust display area based on aspect ratio (landscape vs portrait)
            if aspect_ratio > 1:  # Landscape
                self.image_label.setFixedSize(640, 360)  # Landscape ratio, e.g., 16:9
                frame_resized = cv2.resize(frame, (640, 360))
            else:  # Portrait
                self.image_label.setFixedSize(360, 640)  # Portrait ratio, e.g., 9:16
                frame_resized = cv2.resize(frame, (360, 640))

            # Convert frame to RGB for display
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def capture_image(self):
        # Capture the current frame from the video feed
        ret, frame = self.cap.read()
        if ret:
            # Stop the camera feed temporarily
            self.timer.stop()

            # Determine aspect ratio and resize the frame accordingly
            h, w, _ = frame.shape
            aspect_ratio = w / h
            if aspect_ratio > 1:  # Landscape
                frame_resized = cv2.resize(frame, (640, 360))
            else:  # Portrait
                frame_resized = cv2.resize(frame, (360, 640))

            # Convert and display the captured frame
            rgb_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_image))

            # Save frame temporarily and predict
            img_path = 'captured_dog_image.jpg'
            cv2.imwrite(img_path, frame_resized)  # Save the resized captured image
            breed = predict_breed(img_path)
            self.prediction_label.setText(f"Predicted Breed: {breed}")

            # Restart the camera feed after a short delay (optional)
            QTimer.singleShot(2000, self.start_camera_feed)

    def start_camera_feed(self):
        self.timer.start(30)

    def upload_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Upload Picture", "", "Image Files (*.png *.jpg *.jpeg)", options=options)
        
        if file_path:
            # Stop the camera feed temporarily
            self.timer.stop()

            # Display uploaded image with dynamic aspect ratio
            image = QImage(file_path)
            pixmap = QPixmap.fromImage(image)
            pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            
            # Predict breed
            breed = predict_breed(file_path)
            self.prediction_label.setText(f"Predicted Breed: {breed}")

            # Restart the camera feed after a short delay (optional)
            QTimer.singleShot(2000, self.start_camera_feed)

    def closeEvent(self, event):
        # Release the camera and stop the timer when closing the application
        if self.cap.isOpened():
            self.timer.stop()
            self.cap.release()
        cv2.destroyAllWindows()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DogBreedApp()
    window.show()
    sys.exit(app.exec_())