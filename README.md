# Dog Breed Classification Project

This project focuses on classifying dog breeds using a Convolutional Neural Network (CNN) based on the MobileNetV2 architecture. The implementation is done in Python, leveraging libraries such as TensorFlow and Keras.

## Project Structure

- `_dog_optimized.py`: The main script that contains the CNN model implementation, training logic, and data preprocessing.
- `DogBreedApp.py`: A PyQt5-based graphical user interface for real-time dog breed classification using a webcam or uploaded images.
- `_predict.py`: A script for making predictions on images, including an option to capture images from a webcam.
- `dog_breed_classifier.keras`: The saved trained model (generated after running `_dog_optimized.py`).
- `class_labels.json`: A JSON file containing the mapping between class indices and breed names.
- `README.md`: This file, providing an overview of the project and instructions.
- `requirements.txt`: A list of Python dependencies for the project.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- OpenCV (cv2)
- PyQt5

You can install the required packages using:
```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare the Dataset**: Ensure that your dataset is organized in the `dog-breeds` directory. Each breed should have its own subdirectory containing images of that breed.

2. **Train the Model**: Run the `_dog_optimized.py` script to train the CNN model.
```bash
python _dog_optimized.py
```

3. **Use the GUI Application**: After training, you can use the graphical interface for real-time classification:
```bash
python DogBreedApp.py
```

4. **Make Predictions**: You can also use the `_predict.py` script to make predictions on individual images or capture from a webcam:
```bash
python _predict.py
```

## Model Architecture

The model in `_dog_optimized.py` uses transfer learning with MobileNetV2 as the base model. The architecture includes:

- MobileNetV2 base (pre-trained on ImageNet)
- Global Average Pooling
- Dense layers with dropout for fine-tuning
- Output layer with softmax activation

## GUI Application Features

The `DogBreedApp.py` provides a user-friendly interface with the following features:
- Real-time webcam feed
- Capture image from webcam for classification
- Upload image from file for classification
- Display of predicted dog breed

## Results

The model's performance is evaluated using validation accuracy. After training, the model is saved as `dog_breed_classifier.keras`, and class labels are saved in `class_labels.json`.

## Contributing

Contributions to this project are welcome. Please feel free to submit pull requests or open issues to suggest improvements or report bugs.

## License

This project is open-source and available under the MIT License.

## Acknowledgements

- TensorFlow and Keras documentation
- MobileNetV2 paper and implementation
- Open-source datasets for dog breed images
