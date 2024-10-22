# Dog Breed Multiclass Classification with LSTM

This project focuses on classifying dog breeds using a Long Short-Term Memory (LSTM) neural network. The implementation is done in Python, leveraging libraries such as TensorFlow and Keras.

## Project Structure

- `dog_optimized.py`: The main script that contains the LSTM model implementation and training logic.
- `images/`: Directory containing sample images of different dog breeds used for training and testing.

## Requirements

- Python 3.x
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib

You can install the required packages using:
```bash
pip install tensorflow keras numpy pandas matplotlib
```

## Usage

1. **Prepare the Dataset**: Ensure that your dataset is organized and placed in the `images/` directory. Each breed should have its own subdirectory containing images of that breed.

2. **Run the Script**: Execute the `dog_optimized.py` script to start training the LSTM model.
```bash
python dog_optimized.py
```

3. **Evaluate the Model**: After training, the script will evaluate the model's performance on the test dataset and output the results.

## Model Architecture

The LSTM model in `dog_optimized.py` is designed to handle sequential data and extract temporal features from the input images. The architecture includes:

- Input Layer
- LSTM Layers
- Dense Layers
- Output Layer with Softmax Activation

## Results

The model's performance is evaluated using metrics such as accuracy, precision, and recall. The results are visualized using Matplotlib.

## Contributing

Feel free to fork this repository and contribute by submitting pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License.

## Acknowledgements

- TensorFlow and Keras documentation
- Open-source datasets for dog breed images