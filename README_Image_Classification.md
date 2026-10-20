# Image Classification with TensorFlow and Keras

A binary image-classification project that trains a Convolutional Neural Network (CNN) to distinguish between **Happy** and **Sad** facial expressions/images. The notebook covers image loading and validation, preprocessing, model training, evaluation, single-image inference, and model saving/loading.

> **Note:** This is an educational computer-vision project. Predictions are not a psychological assessment and should not be used to infer a person's mental health or emotional state.

## Project Overview

The model takes an image as input, resizes it to `256 × 256` pixels, scales pixel values to the `[0, 1]` range, and predicts one of two classes: `Happy` or `Sad`.

The implementation uses TensorFlow/Keras to build a sequential CNN with convolutional and max-pooling layers, followed by dense layers and a sigmoid output.

## Dataset

The notebook loads images from a local directory named `data` using `tf.keras.utils.image_dataset_from_directory`.

Expected directory layout:

```text
data/
├── Happy/
│   ├── image_001.jpg
│   └── ...
└── Sad/
    ├── image_001.jpg
    └── ...
```

The saved notebook output reports **305 images across 2 classes**. The dataset source and usage permissions are not documented in the notebook; add the original source and license here before publishing the dataset or redistributing images.

Supported image extensions checked by the notebook include `.jpeg`, `.jpg`, `.bmp`, and `.png`.

## Model Architecture

The CNN is implemented with `tf.keras.Sequential` and includes:

- `Conv2D` layers with ReLU activation
- `MaxPooling2D` layers for spatial downsampling
- `Flatten` to convert feature maps into a vector
- A dense hidden layer with 256 units and ReLU activation
- A single sigmoid output unit for binary classification

**Training configuration**

- Input size: `256 × 256 × 3`
- Optimizer: Adam
- Loss: Binary cross-entropy
- Metric during training: accuracy
- Epochs: 20
- TensorBoard callback for training logs

## Reported Evaluation

The notebook's saved test-set output reports:

| Metric | Reported value |
|---|---:|
| Binary accuracy | 84.31% |
| Precision | 100.00% |
| Recall | 75.00% |

These values are copied from the notebook output and have not been independently reproduced. The test set appears to be small, so the metrics may vary substantially with a different split. Treat them as preliminary results rather than evidence of general performance.

## Technologies

- Python
- TensorFlow / Keras
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- Jupyter Notebook

## Getting Started

### 1. Clone the repository

Replace the example URL below with the actual repository URL if the repository name or owner differs.

```bash
git clone https://github.com/AmirMohammadNajafi1383/Image_classification.git
cd Image_classification
```

### 2. Create an environment and install dependencies

```bash
python -m venv .venv
```

Activate it:

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows**
```powershell
.venv\Scripts\activate
```

Install the required packages:

```bash
python -m pip install tensorflow opencv-python matplotlib numpy jupyter
```

### 3. Add the dataset

Place the image folders under `data/` using the directory layout shown above. The folder names determine the class labels assigned by `image_dataset_from_directory`.

### 4. Run the notebook

```bash
jupyter notebook image_classifiction.ipynb
```

Run the notebook cells in order. If your notebook has a different filename, use that filename in the command.

## Model Inference and Saving

The notebook demonstrates how to:

1. Read an image with OpenCV.
2. Resize it to `256 × 256`.
3. Normalize pixel values by dividing by `255`.
4. Run `model.predict()` to obtain a sigmoid score.
5. Save and reload the model from `models/imageclassifier.h5`.

Ensure the `models/` directory exists before saving the model. For new projects, the native Keras `.keras` format is generally preferable to the legacy HDF5 format.

## Limitations and Recommended Improvements

- **Reproducibility:** Set random seeds and document the exact dataset source, class counts, and train/validation/test split.
- **Evaluation:** Report a confusion matrix and per-class precision, recall, and F1-score. Use a larger independent test set where possible.
- **Data splitting:** Split data at the image or subject level as appropriate, and ensure near-duplicate or related images do not appear across training and test sets.
- **Model generalization:** Compare against a transfer-learning baseline such as MobileNetV2 or EfficientNet, and use data augmentation only on the training set.
- **Training stability:** Consider early stopping and saving the best checkpoint based on validation performance.
- **Inference clarity:** Document the class-index mapping produced by the dataset loader, and convert the sigmoid score into a class label consistently.
- **Dependencies:** Avoid installing `tensorflow-gpu` separately in modern TensorFlow environments unless the selected TensorFlow version specifically requires it.

## Repository Structure

```text
Image_classification/
├── image_classifiction.ipynb
├── data/                       # Add locally; dataset may not be included
└── models/
    └── imageclassifier.h5      # Generated when the save cell runs
```

## Author

**Amir Mohammad Najafi**

GitHub: [AmirMohammadNajafi1383](https://github.com/AmirMohammadNajafi1383)
