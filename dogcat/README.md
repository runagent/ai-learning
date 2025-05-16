# Dog vs Cat Image Classification with PyTorch

This project implements a simple CNN model for classifying images of dogs and cats using PyTorch.

## Project Structure

- `utils.py`: Contains utility functions for data loading and preprocessing
- `model.py`: Contains the CNN model architecture
- `train.py`: Contains the training code
- `evaluate.py`: Contains code for evaluating the model on test data

## Dataset

The dataset should be organized as follows:

```
/path/to/dataset/
├── train/
│   ├── cats/
│   │   ├── cat.0.jpg
│   │   ├── cat.1.jpg
│   │   └── ...
│   └── dogs/
│       ├── dog.0.jpg
│       ├── dog.1.jpg
│       └── ...
└── test/
    ├── cats/
    │   ├── cat.0.jpg
    │   ├── cat.1.jpg
    │   └── ...
    └── dogs/
        ├── dog.0.jpg
        ├── dog.1.jpg
        └── ...
```

## Requirements

- Python 3.6+
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn
- tqdm
- seaborn

You can install the required packages using pip:

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm seaborn
```

## Usage

### Training

To train the model, run:

```bash
python train.py
```

This will:
1. Load and preprocess the data
2. Create a CNN model
3. Train the model for 10 epochs
4. Save the trained model to `dogcat_model.pth`
5. Plot and save the training history to `training_history.png`

### Evaluation

To evaluate the trained model on the test set, run:

```bash
python evaluate.py
```

This will:
1. Load the trained model from `dogcat_model.pth`
2. Evaluate the model on the test set
3. Plot and save the confusion matrix to `confusion_matrix.png`
4. Print the classification report
5. Visualize some predictions and save them to `predictions.png`

### Prediction

To use the trained model to predict the class of a single image, run:

```bash
python predict.py path/to/your/image.jpg
```

You can also specify a different model path:

```bash
python predict.py path/to/your/image.jpg --model_path path/to/your/model.pth
```

This will:
1. Load the trained model
2. Preprocess the input image
3. Make a prediction (cat or dog)
4. Display the image with the prediction

## Model Architecture

The model is a simple CNN with the following architecture:

- 4 convolutional blocks, each consisting of:
  - Convolutional layer
  - Batch normalization
  - ReLU activation
  - Max pooling
- Fully connected layers
- Dropout for regularization

## Data Preprocessing

The data preprocessing includes:

- Resizing images to 224x224
- Data augmentation for training (random horizontal flip, rotation, color jitter)
- Normalization using ImageNet mean and standard deviation

## Results

After training, the model should achieve an accuracy of around 90-95% on the test set. The exact results will be displayed when running `evaluate.py`.
