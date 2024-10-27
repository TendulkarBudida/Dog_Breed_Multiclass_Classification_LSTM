import os
import numpy as np
import tensorflow as tf
import json
import cv2
from keras.preprocessing.image import img_to_array, load_img

# Disable OneDNN optimizations for TensorFlow
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress TensorFlow deprecation warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Suppress TensorFlow warnings
tf.get_logger().setLevel('ERROR')

# Load the trained model
model = tf.keras.models.load_model('dog_breed_classifier.keras')

# Load the class labels
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)
breed_names = {v: k for k, v in class_labels.items()}  # Reverse dictionary to map index to breed name

# Define image size expected by the model
img_size = (224, 224)

def predict_breed(image_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=img_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Rescale as done during training
    
    # Predict the breed
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)[0]
    
    # Get the breed name
    predicted_breed = breed_names[predicted_class]
    
    return predicted_breed


# Capture image from the camera (optional)
def capture_and_predict():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Camera not found.")
        return

    # Create the window and set the initial size to 500x500
    cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera', 500, 500)

    print("Press 'c' to capture an image, or 'q' to quit.")
    while True:
        ret, frame = cap.read()
        cv2.imshow('Camera', frame)

        # Check if window is still open
        if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
            print("Window closed manually. Exiting...")
            break

        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Capture and save the image, then predict
            img_path = 'captured_dog_image.jpg'
            cv2.imwrite(img_path, frame)  # Save captured image
            breed = predict_breed(img_path)  # Predict the breed
            print(f'Predicted Dog Breed: {breed}')
        elif key == ord('q'):
            # Quit the loop if 'q' is pressed
            print("Exiting...")
            break

    # Release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()




# Uncomment the line below to capture from camera and predict
capture_and_predict()

# Or predict from a specific file path
# image_path = 'hehe.png'
# breed = predict_breed(image_path)
# print(f'Predicted Dog Breed: {breed}')