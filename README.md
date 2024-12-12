# COMP433-Project-Group-E


## Tissue Segmentation in Digital Pathology Preview Images

This project implements deep learning models for automated segmentation of regions of interest (ROIs) in preview images within digital pathology scanning. The goal is to enhance the efficiency and accuracy of tissue analysis by automating the identification of tissue regions in Whole Slide Images (WSIs), leveraging deep learning architectures.

## Overview

By implementing and comparing different deep learning architectures such as U-Net, U-Net++ with attention mechanisms, and incorporating encoders like EfficientNet, this project aims to determine the most effective approach for tissue segmentation in preview images, addressing challenges like complex tissue structures, varying image quality, and class imbalance.

## Implementation Details

This builds upon advanced deep learning techniques for medical image segmentation, involving:

- **Data Preprocessing and Augmentation**: Normalization, resizing, and various augmentation techniques to enhance model robustness.
- **Deep Learning Models**: Implementing and training architectures like U-Net, U-Net++ with SCSE attention, as well as utilizing EfficientNet encoders.
- **Transfer Learning**: Utilizing pre-trained encoders to improve feature extraction.
- **Loss Functions and Metrics**: Employing Dice Loss for optimization and Intersection over Union (IoU) for evaluation.
- **Comparison and Evaluation**: Assessing model performance to identify the most effective architecture.

### Key Components

- **Data Augmentation Techniques**:
  - **Random Rotations and Flips**: To simulate varied tissue orientations.
  - **Zooming and Scaling**: To handle different tissue sizes.
  - **Brightness Adjustment and Gaussian Noise**: To mimic variations in image quality.
- **Deep Learning Architectures**:
  - **U-Net**: A CNN with contracting and expanding paths for precise localization.
  - **U-Net++ with SCSE Attention**: Enhanced U-Net with nested skip connections and attention mechanisms.
  - **EfficientNet**: Utilizes compound scaling for improved performance with reduced computational cost.
- **Optimization**:
  - **Optimizer**: AdamW with a learning rate of 1e-4.
  - **Loss Function**: Dice Loss, suitable for handling class imbalance.
- **Evaluation Metrics**:
  - **Intersection over Union (IoU)**: Measures overlap between predicted masks and ground truth.

## Data Source

This project utilizes a dataset of 17,375 preview images provided by **Huron Digital Pathology**. Each image is 512×512 pixels and includes a corresponding pixel-level annotated tissue mask (ground truth). The dataset encompasses diverse tissue samples with varying characteristics such as size, shape, texture, and artifacts like pen marks and fading.

### Data Preparation

- **Normalization**: Images are normalized using standard ImageNet statistics.
- **Resizing**: Images are resized to 256×256 pixels to meet model input requirements and optimize computational efficiency.
- **Augmentation**: Techniques like rotations, flips, zooming, brightness adjustments, and adding Gaussian noise are applied to enhance dataset diversity and model robustness.

## Getting Started

### Prerequisites

To run this project locally, ensure the following are installed:

- **Python 3.8+**
- **Jupyter Notebook**
- **PyTorch**
- **Torchvision**
- **Segmentation Models PyTorch (smp)**
- **Additional Libraries**: `numpy`, `pandas`, `matplotlib`, `scikit-learn`

Install the necessary packages via:

```bash
pip install torch torchvision
pip install segmentation-models-pytorch
pip install numpy pandas matplotlib scikit-learn
```

### Running the Project

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/nlagarde15/COMP433-GroupE.git
   cd to be updated.
   ```

2. **Prepare the Data**:

   - Download the dataset provided by Huron Digital Pathology.
   - Organize the dataset into `train`, `validation`, and `test` directories, each containing `images` and `masks` subdirectories.

3. **Run the Jupyter Notebook**:

   - Launch Jupyter Notebook:

     ```bash
     jupyter notebook
     ```

   - Steps to be updated.

4. **Experiment with Different Models**:

   - Modify the notebook to switch between different architectures and encoders.
   - Adjust hyperparameters like learning rate, batch size, and number of epochs to improve performance.

## Results

Out of the eight architectures initially tested, the following three achieved the highest IoU scores before final refinements:

- **U-Net++**: IoU of 0.9118
- **U-Net++ with SCSE Attention**: IoU of 0.9089
- **FPN**: IoU of 0.9050

After optimization and hyperparameter tuning involving different encoders and loss functions, the best-performing configuration was:

- **U-Net++ with EfficientNet-B4 Encoder and Dice + BCE Loss**:
  - Achieved an IoU of **93.38%**, demonstrating a significant improvement over the past results.
 
## Running the model

- The trained model can be downloaded from the [Initial Release](https://github.com/nlagarde15/COMP433-GroupE/releases/tag/v1.0.0)
and tested by running [test_best_model](test_best_model.ipynb). 
- The test data with sliced images and slices masks can be download from [test data](https://github.com/nlagarde15/COMP433-GroupE/tree/main/Test%20Dataset)

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE.txt) file for more details.

 
