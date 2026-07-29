# CIFAR-10 Image Classification with PyTorch

A reproducible CIFAR-10 image classification project implemented with PyTorch.

The project supports three model architectures:

- a custom CNN implemented from scratch
- ResNet-18 adapted for CIFAR-10
- VGG-16 with Batch Normalization

Users can select the model through a command-line argument.

## Overview

CIFAR-10 is an image classification dataset containing 60,000 color images across 10 classes:

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
- ResNet-18 adapted for 32 × 32 images
- VGG-16 with Batch Normalization
- Model selection through command-line arguments
- Automatic CIFAR-10 download
- Training and evaluation loops
- GPU acceleration with CUDA
- Data augmentation
  - Random crop
  - Random horizontal flip
  - Random erasing
- CIFAR-10-specific input normalization
- Training history recording
- Accuracy and loss curve visualization with Matplotlib

## Project Structure

```text
cifar10-classification-pytorch/
├── dataset.py       # Dataset, preprocessing, and DataLoader creation
├── models.py        # Custom CNN and torchvision-based model definitions
├── trainer.py       # Training and evaluation functions
├── visualize.py     # Accuracy and loss curve visualization
├── main.py          # Main training script and command-line interface
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
- Matplotlib
- Git

An NVIDIA GPU is optional. The program automatically uses CUDA when a CUDA-enabled GPU is available and otherwise runs on the CPU.

### Recommended GPU environment

The project has been developed with an NVIDIA GeForce RTX 4060.

The custom CNN and ResNet-18 can generally be trained on consumer GPUs with modest VRAM. VGG-16 is significantly larger and may require more memory or a smaller batch size depending on the available GPU.

CPU training is supported, although it will be slower.

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

For NVIDIA GPU execution, install a CUDA-enabled version of PyTorch that matches your environment. Use the command provided by the official PyTorch installation selector:

```text
https://pytorch.org/get-started/locally/
```

A compatible NVIDIA driver is required for CUDA execution.

### 4. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

## Verify the GPU Environment

Run:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Example output with an NVIDIA GPU:

```text
PyTorch: 2.x.x+cuXXX
CUDA available: True
GPU: NVIDIA GeForce RTX 4060
```

## Usage

### Custom CNN

The custom CNN is used by default:

```bash
python main.py
```

It can also be selected explicitly:

```bash
python main.py --model_name cnn
```

### ResNet-18

```bash
python main.py --model_name resnet18
```

### VGG-16

```bash
python main.py --model_name vgg16
```

To display the available command-line options:

```bash
python main.py --help
```

Available models:

| Argument | Architecture |
|---|---|
| `cnn` | Custom CNN |
| `resnet18` | ResNet-18 adapted for CIFAR-10 |
| `vgg16` | VGG-16 with Batch Normalization |

When `--model_name` is omitted, the default model is `cnn`.

On the first run, torchvision automatically downloads the CIFAR-10 dataset into:

```text
./data
```

During training, the program displays the loss and accuracy for both the training and test datasets. After training, the recorded history is visualized with Matplotlib and figures can be saved in the `outputs/` directory.

## Training Configuration

The current default configuration is defined in `main.py` and `dataset.py`.

```python
num_epochs = 50
n_hidden = 128
learning_rate = 0.0001
batch_size = 32
```

To perform a quick execution test, temporarily change:

```python
num_epochs = 50
```

to:

```python
num_epochs = 1
```

## Supported Models

### Custom CNN

The custom CNN contains six convolutional layers, three max-pooling layers, dropout, and a fully connected classifier.

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

### ResNet-18

The ResNet-18 implementation is based on `torchvision.models.resnet18`.

Because CIFAR-10 images are only 32 × 32 pixels, the original ImageNet-style input stage is modified:

- the first convolution uses a 3 × 3 kernel with stride 1
- the initial max-pooling layer is removed
- the final fully connected layer outputs 10 classes

The model is trained from scratch with `weights=None`.

### VGG-16

The VGG model is based on `torchvision.models.vgg16_bn`.

- the network includes Batch Normalization
- the final classification layer is replaced with a 10-class output layer
- the model is trained from scratch with `weights=None`

VGG-16 is substantially larger than the custom CNN and ResNet-18, so it requires more memory and training time.

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

The current version focuses on model selection and the basic training pipeline.

The following features are planned for future development:

- Saving trained model weights
- Confusion matrix visualization
- Class-wise accuracy
- Misclassified image visualization
- Command-line options for epochs, batch size, and learning rate
- Reproducible random seed settings
- Automated tests
- Benchmarking all supported architectures

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details.
