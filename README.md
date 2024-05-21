# ASL Gesture Recognition

## Project Overview

This project aims to develop a system that can recognize American Sign Language (ASL) gestures using machine-learning models. The project includes data collection, preprocessing, model training, and real-time gesture recognition.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Collection](#dataset-collection)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Comparison](#model-comparison)
- [Real-Time Inference](#real-time-inference)
- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Dataset Collection

The dataset is collected using a webcam. The images are augmented using the `ImageDataGenerator` from Keras to increase the variability of the dataset. Each class represents a different ASL gesture.

1. **Initialize Data Directory**: Creates a directory to store the data if it doesn't exist.
2. **Data Augmentation**: Utilizes `ImageDataGenerator` for augmenting the collected images.
3. **Collect Data from Webcam**: Captures images from the webcam, augments them, and saves them in class-specific directories.

## Data Preprocessing

After collecting the data, we process the images to extract hand landmarks using MediaPipe. The coordinates of these landmarks are normalized and stored for training the models.

1. **Load Images**: Reads images from the data directories.
2. **Extract Landmarks**: Uses MediaPipe to extract hand landmarks.
3. **Normalize Coordinates**: Normalizes the landmark coordinates.
4. **Save Processed Data**: Stores the processed data in a pickle file for further use.

## Model Training

Two types of models are trained to recognize ASL gestures: Random Forest and Convolutional Neural Network (CNN).

1. **Random Forest Model**: 
    - Loads the processed data.
    - Splits the data into training and testing sets.
    - Trains a Random Forest classifier.
    - Saves the trained model.

2. **CNN Model**: 
    - Reshapes the data for CNN input.
    - Creates and trains a CNN model.
    - Evaluate the model and save it.

## Model Comparison

The performance of the trained models is compared by calculating the accuracy on the test set.

1. **Random Forest Evaluation**: Loads the Random Forest model and evaluates its accuracy.
2. **CNN Evaluation**: The accuracy of the CNN model is printed during training.

## Real-Time Inference

The trained Random Forest model is used for real-time gesture recognition using webcam input.

1. **Setup MediaPipe Hands**: Initializes MediaPipe for hand tracking.
2. **Load Model**: Loads the trained Random Forest model.
3. **Real-Time Prediction**: Captures frames from the webcam, processes them to extract hand landmarks, and uses the model to predict the ASL gesture.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/asl-gesture-recognition.git
    cd asl-gesture-recognition
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Collect Data**:
    ```bash
    python collect_data.py
    ```

2. **Preprocess Data**:
    ```bash
    python preprocess_data.py
    ```

3. **Train Models**:
    ```bash
    python train_models.py
    ```

4. **Compare Models**:
    ```bash
    python compare_models.py
    ```

5. **Real-Time Inference**:
    ```bash
    python real_time_inference.py
    ```

## Dependencies

- Python 3.7+
- OpenCV
- TensorFlow / Keras
- Scikit-Learn
- MediaPipe
- Matplotlib
- NumPy
- Pickle

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
