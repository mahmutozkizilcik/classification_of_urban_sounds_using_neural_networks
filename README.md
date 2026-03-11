# AIN313 Assignment 3: Classification of Urban Sounds using Neural Networks

**Student:** Mahmut Özkızılcık  
**Student ID:** 2220765019  
**Course:** AIN313 — Introduction to Artificial Intelligence

## Overview

This project implements and compares two different neural network architectures for classifying urban environmental sounds using the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset:

1. **Multi-Layer Perceptron (MLP)** — built from scratch using only NumPy
2. **Convolutional Neural Network (CNN)** — implemented with PyTorch

Audio files are converted to **Mel Spectrograms** and used as input features for both models. The project includes data preprocessing, feature extraction, gradient checking, hyperparameter search, training, visualization, and a detailed comparison of all models.

## Project Structure

```
├── Assigment 3.ipynb        # Main Jupyter Notebook (with outputs & visualizations)
├── code/
│   └── Assigment 3.py       # Python script version of the notebook
├── AIN313_Assignment_3.pdf  # Assignment description
├── requirements.txt         # Python dependencies
├── .gitignore               # Git ignore rules
└── README.md                # This file
```

## Dataset

This project uses the **UrbanSound8K** dataset, which contains 8,732 labeled sound excerpts (≤4s) of urban sounds from 10 classes:

| Class ID | Sound |
|----------|-------|
| 0 | Air Conditioner |
| 1 | Car Horn |
| 2 | Children Playing |
| 3 | Dog Bark |
| 4 | Drilling |
| 5 | Engine Idling |
| 6 | Gun Shot |
| 7 | Jackhammer |
| 8 | Siren |
| 9 | Street Music |

### Download

1. Download the dataset from [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html)
2. Extract it into a `dataset/` folder in the project root:
   ```
   dataset/
   ├── UrbanSound8K.csv
   ├── fold1/
   ├── fold2/
   ...
   └── fold10/
   ```

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

> **Note:** For GPU support with PyTorch, install the appropriate CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).

## Usage

### Jupyter Notebook (Recommended)

```bash
jupyter notebook "Assigment 3.ipynb"
```

### Python Script

```bash
python code/"Assigment 3.py"
```

## Implementation Details

### Part 1 — Feature Extraction
- Audio loaded with `librosa` (kaiser_fast resampling)
- 128-bin Mel Spectrograms computed and converted to dB scale
- Padded/truncated to fixed width of 174 time frames
- Data split: 80% train / 20% test with stratified sampling
- Global min-max normalization (fitted on training set only)

### Part 2 — MLP (From Scratch with NumPy)
- Fully custom implementation: forward pass, backpropagation, gradient descent
- ReLU activation (hidden layers) + Softmax (output layer)
- He weight initialization
- Cross-entropy loss
- Gradient checking to verify backpropagation correctness
- Grid search over architectures, learning rates, and batch sizes
- Learning rate decay

### Part 3 — CNN (PyTorch)
- **CNN_1Layer:** 1 Conv block (16 filters) → MaxPool → FC
- **CNN_2Layer:** 2 Conv blocks (32 → 64 filters) → MaxPool → 2 FC layers
- Adam optimizer with exponential LR scheduler
- Trained on GPU (CUDA)

### Part 4 — Model Comparison
- Bubble chart visualization comparing accuracy vs. model complexity
- Detailed analysis of each model's strengths and weaknesses

## Results

| Model | Test Accuracy |
|-------|--------------|
| MLP (1 Hidden Layer, 128 neurons) | ~55% |
| MLP (2 Hidden Layers, 128→64) | ~46% |
| CNN (1 Layer) | ~80% |
| **CNN (2 Layers)** | **~84%** |

**Conclusion:** The 2-Layer CNN significantly outperforms all MLP variants, demonstrating the superiority of convolutional architectures for spectrogram-based audio classification.

## Technologies

- Python 3
- NumPy, Pandas
- Matplotlib, Seaborn
- Librosa
- PyTorch
- scikit-learn
