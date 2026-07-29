# CIFAR-10 Image Classification with PyTorch

A simple and reproducible image classification project using a custom convolutional neural network (CNN) implemented in PyTorch.

This project trains a CNN from scratch on the CIFAR-10 dataset and evaluates its classification accuracy on the test set.

## Overview

CIFAR-10 is an image classification dataset containing 60,000 color images across 10 classes.

The classes are:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

Each image has the following shape:

```text
3 × 32 × 32
```

where `3` represents the RGB color channels.

## Features

- Custom CNN implemented with PyTorch
- Automatic CIFAR-10 download
- Training and evaluation loops
- GPU acceleration with CUDA
- Data augmentation
  - Random crop
  - Random horizontal flip
  - Random erasing
- CIFAR-10-specific input normalization
- Training history recording

## Project Structure

```text
cifar10-classification-pytorch/
├── dataset.py       # Dataset, preprocessing, and DataLoader creation
├── models.py        # CNN architecture
├── trainer.py       # Training and evaluation functions
├── main.py          # Main training script
├── requirements.txt # Python dependencies
├── README.md
├── LICENSE
└── .gitignore
```

## Requirements

- Python 3.10 or later
- PyTorch
- torchvision
- NumPy
- tqdm
- Git

An NVIDIA GPU is optional.

The program automatically uses CUDA when a CUDA-enabled GPU is available. Otherwise, it runs on the CPU.

### Recommended GPU environment

The project has been developed with an NVIDIA GeForce RTX 4060.

A powerful GPU is not required because CIFAR-10 images are small. GPUs with approximately 4 GB or more of VRAM should generally be sufficient with the default batch size.

CPU training is also supported, although it will be slower.

## Installation

### 1. Clone the repository

Open PowerShell, Command Prompt, or a terminal and run:

```bash
git clone https://github.com/Manstein0704/cifar10-classification-pytorch.git
cd cifar10-classification-pytorch
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

When the environment is activated, the terminal should display:

```text
(.venv)
```

If PowerShell blocks the activation script, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
```

#### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install PyTorch

For CPU execution, run:

```bash
pip install torch torchvision
```

For NVIDIA GPU execution, install a CUDA-enabled version of PyTorch that matches your environment.

Use the installation command provided by the official PyTorch installation selector:

```text
https://pytorch.org/get-started/locally/
```

You do not normally need to install the full CUDA Toolkit separately when using the official PyTorch wheels. However, a compatible NVIDIA driver is required.

### 4. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

## Verify the GPU Environment

Run the following command:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Example output with an NVIDIA GPU:

```text
PyTorch: 2.x.x+cuXXX
CUDA available: True
GPU: NVIDIA GeForce RTX 4060
```

When CUDA is unavailable, the program automatically uses the CPU.

## Usage

Start training with:

```bash
python main.py
```

On the first run, torchvision automatically downloads the CIFAR-10 dataset into:

```text
./data
```

During training, the program displays the loss and accuracy for both the training and test datasets.

Example:

```text
cuda:0
Epoch [1/50] Train Loss: 1.8500 Train Acc: 31.20% | Test Loss: 1.6200 Test Acc: 40.10%
```

The exact results depend on random initialization, hardware, and training settings.

## Training Configuration

The current default configuration is defined in `main.py`.

```python
num_epochs = 50
n_hidden = 128
learning_rate = 0.0001
batch_size = 32
```

To perform a quick execution test, change:

```python
num_epochs = 50
```

to:

```python
num_epochs = 1
```

## Data Augmentation

The training images are augmented using:

```python
transforms.RandomCrop(32, padding=4)
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomErasing(p=0.3)
```

These transformations help reduce overfitting by exposing the model to slightly modified versions of the training images.

The test images are not augmented.

## Input Normalization

The RGB channels are normalized using statistics calculated from the CIFAR-10 training dataset:

```python
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
```

Normalization helps stabilize neural network training by placing the input values on comparable scales.

## Model Architecture

The model is a custom CNN trained from scratch.

Its main structure is:

```text
Input image
    ↓
Convolution layers: 3 → 32 channels
    ↓
Max pooling and dropout
    ↓
Convolution layers: 32 → 64 channels
    ↓
Max pooling and dropout
    ↓
Convolution layers: 64 → 128 channels
    ↓
Max pooling and dropout
    ↓
Fully connected classifier
    ↓
10-class output
```

No pretrained model or pretrained weights are used.

## Dataset

CIFAR-10 contains:

| Split | Number of images |
|---|---:|
| Training | 50,000 |
| Test | 10,000 |
| Total | 60,000 |

The dataset is downloaded automatically through `torchvision.datasets.CIFAR10`.

The dataset files are stored locally and are not committed to this repository.

## Current Limitations

The current version focuses on the basic training pipeline.

The following features are planned for future development:

- Saving trained model weights
- Plotting learning curves
- Confusion matrix visualization
- Class-wise accuracy
- Misclassified image visualization
- Configuration file support
- Reproducible random seed settings
- Automated tests
- Comparison with ResNet models

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details.
