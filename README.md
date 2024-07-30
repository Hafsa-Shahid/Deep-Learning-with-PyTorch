# Image Segmentation with PyTorch using U-Net Architecture:

This repository contains a guided project on image segmentation using the U-Net architecture with PyTorch, completed with Coursera. The project demonstrates the implementation of a deep learning model to perform image segmentation tasks.

## Overview:

Image segmentation is a crucial task in computer vision, involving the partitioning of an image into multiple segments or regions. The U-Net architecture, introduced in 2015 for biomedical image segmentation, is known for its ability to yield precise segmentation results with limited training data.

## Contents

- **notebook/:** Jupyter notebook containing the project code and experiment
- **data/:** [Sample dataset used for training and testing](https://github.com/VikramShenoy97/Human-Segmentation-Dataset)
- **README.md:** Project documentation (this file)

## Prerequisites

Before running the code, ensure you have the following dependencies installed:

- Python 3.6+
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-image
- jupyter

You can install the required packages using the following command:

            !pip install torch torchvision numpy matplotlib scikit-image jupyter
  
## Getting Started

1. Clone this repository:
   ```bash
   https://github.com/Hafsa-Shahid/Deep-Learning-with-PyTorch.git
   
2. Access the notebook in the respective folder.
  
3. Get the data from [here](https://github.com/VikramShenoy97/Human-Segmentation-Dataset)

## U-Net Architecture

The U-Net architecture consists of an encoder-decoder structure:

Encoder: Captures context using convolutional layers and downsampling.

Decoder: Enables precise localization using upsampling and concatenation with high-resolution features from the encoder.

Here we have used efficientnet encoder and the weights of imagenet dataset, and these can be experimented for different scenarios.

## About the Project

 - Wrote a custom dataset class for Image-mask dataset. Then, applied augmentation to augment images as well as its masks using the albumentation library.

 - Loaded a pretrained state of the art convolutional neural network for segmentation problem(U-Net) using segmentation model pytorch library. 

 - Created train and evaluator function which was helpful to write training loop and used the training loop to train the model.

## Results

The model achieves good accuracy in segmenting images from the given dataset. Sample result is shown below:

![image](https://github.com/user-attachments/assets/cc65137c-253c-4de0-8e31-22caf282164b)

## Contributions 
  
  Contributions are welcome! If you have any suggestions or improvements, feel free to open an issue or submit a pull request. Also, you can build upon this notebook and move it towards deployment and beyond as required.

## Acknowledgments

 - Coursera and the [instructor](https://github.com/parth1620) for the guided project.
 - Authors of the U-Net architecture.
 - PyTorch community for the excellent deep learning framework.
