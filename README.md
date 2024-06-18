# Low-Light-Image-Enhancement-Using-CNN

## Project Overview

This project focuses on enhancing images taken in low-light conditions using a Convolutional Neural Network (CNN). The objective is to develop a model that effectively denoises and enhances these images, improving their clarity and overall quality.

## Project Goals

- Design a CNN architecture specifically for image denoising.
- Train the model on a dataset of paired low-light and high-light images.
- Evaluate the model's performance using metrics such as Peak Signal-to-Noise Ratio (PSNR).
- Deploy the model and demonstrate its ability to enhance low-light images.

## System Architecture and Data Flow

### System Components

- **Dataset:** Paired images of low-light and high-light conditions.
- **Model Architecture:** A CNN designed for noise reduction and image enhancement.
  - Input Layer
  - Convolutional Layers with ReLU activation
  - Skip Connections
  - Output Layer with sigmoid activation

### Data Flow

1. Loading Data
2. Training the Model
3. Model Testing
4. Image Enhancement

## Techniques and Methodologies

### CNN Architecture Details

- **Layers:** Several convolutional layers with ReLU activation, interspersed with skip connections.
- **Activation Functions:** ReLU in hidden layers and sigmoid in the output layer.
- **Loss Function:** Mean Squared Error (MSE).
- **Metrics:** Peak Signal-to-Noise Ratio (PSNR).

### Data Pre-Processing

- Normalization: Pixel values normalized to [0,1].
- Shape Adjustment: Input images adjusted to match the model's expected input shape.
- Channel Handling: Grayscale images converted to RGB.

### Training and Evaluation

- **Training Process:** Model trained for 22 epochs using Google Colab.
- **Evaluation Metrics:** PSNR calculated to assess denoising performance.

## Implementation Pipeline

### Data Collection and Storage

- **Sources:** Dataset provided on Slack.
- **Storage Method:** Images uploaded to Google Drive and stored in directories within Google Colab.
- **Schema Design:** Organized into separate folders for low-light and high-light images.

### Data Processing

- **Loading and Preprocessing:** Images loaded using `imageio` library.
- **Normalization:** Pixel values normalized to [0,1].

### Model Development

- **Model Selection:** CNN architecture chosen for its effectiveness in image processing tasks.
- **Inspiration:** Architecture inspired by a Kaggle notebook using parallel networks and complex CNN architectures.
- **Training:** Model trained using Adam optimizer and MSE loss function.

### Deployment and Monitoring

- **Tools and Platforms:** Model trained and deployed on Google Colab.
- **Deployment Process:** Model saved in HDF5 format.
- **Monitoring:** Model performance monitored using PSNR values during testing and inference.

## Results

The average PSNR value achieved was 27.06695, indicating significant improvement in image quality.

For reference, the model's architecture and further details can be found at [Kaggle Notebook](https://www.kaggle.com/code/basu369victor/low-light-image-enhancement-with-cnn).

## Usage Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/krish-sky1ark/Low-Light-Image-Enhancement-Using-CNN

2. Upload the dataset mentioned in main.ipynb file to Google Drive.
3. Change the directory to
   ```bash
   cd Low-Light-Image-Enhancement-Using-CNN
5. Run the main.ipynb file
   ```bash
   jupyter notebook main.ipynb
7. Comment out the model formation and training part in case you want to use the pre-trained model denoising_model.h5
8. Train the Model and use the trained model to enhance low-light images.
