# Ultrasound Segmentation with Depth-Aware Positional Encoding

This repository contains the implementation of a novel **depth-aware positional encoding** and **spacing-aware attention mechanism** for ultrasound image segmentation. The project leverages a modified UNet++ architecture to achieve high-accuracy segmentation of skin tissues, including the epidermis, dermis, and subcutaneous fat layers.

## Features
- **Depth-Aware Positional Encoding:** Dynamically encodes spatial depth information to enhance segmentation accuracy.
- **Spacing-Aware Attention Mechanism:** Integrates spatial encoding to improve focus on tissue regions of interest.
- **Custom Loss Function:** Combines Binary Cross-Entropy (BCE) and Dice Loss for robust training.
- **Data Augmentation Pipeline:** Supports flipping, Gaussian noise, affine transformations, and brightness adjustments to improve model generalization.
- **Pre-trained Backbones:** Utilizes various encoders (e.g., MobileNetV2, ResNet34, Xception) with flexible configurations.

## Key Components
- **Segmentation Model:** Implements a modified UNet++ backbone with attention mechanisms.
- **Positional Encoding Module:** Generates 2D sine-cosine positional encodings for spatial context.
- **Custom Dataset Loader:** Handles ultrasound images and metadata for training and evaluation.

## Results
The method was evaluated on a test set, achieving improved Dice scores across all tissue types when compared to models without depth-aware encoding. Detailed results can be found in the corresponding [paper](#).

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/ultrasound-segmentation.git](https://github.com/dteamsz/Spatial-Aware-Attention.git
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### Training
Train the model with a specific encoder (e.g., MobileNetV2):
```bash
python train_depthawarev3.py --encoder mobilenet_v2
