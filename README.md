# 🧠 Brain Tumor Segmentation using Attention U-Net

## Overview

This project demonstrates a deep learning approach to accurately segment brain tumors from MRI scans using an **Attention U-Net** architecture. The model focuses on improving segmentation accuracy by employing attention gates that enhance its ability to focus on important regions of the input images. The combination of **Attention U-Net** with advanced **data augmentation** techniques and **custom loss functions** like the **Dice coefficient** ensures robust performance, even in the presence of class imbalance and small segmentation regions.

## Key Features
- **Attention U-Net Architecture**: Utilizes attention gates to refine the segmentation process by selectively focusing on relevant features.
- **Custom Loss Function**: Incorporates a combination of **binary crossentropy** and the **Dice coefficient**, ensuring precise segmentation, especially for small and imbalanced regions.
- **Data Augmentation**: Applies real-time augmentation techniques, including rotation, zoom, flipping, and shifts, to improve the model's generalization ability.
- **Callback Optimization**: Employs **EarlyStopping**, **ModelCheckpoint**, and **ReduceLROnPlateau** callbacks to prevent overfitting and optimize training.
- **Model Performance Visualization**: Provides detailed visualization of training history and model predictions, showcasing the ability to segment tumor regions accurately.

## Project Motivation

The motivation behind this project stems from the increasing need for accurate and efficient brain tumor detection methods. Manual tumor segmentation is time-consuming and prone to human error. Automating this process using deep learning can significantly reduce diagnostic time and improve patient outcomes.

This project aims to address:
- **Accuracy**: Enhancing segmentation accuracy through attention mechanisms.
- **Efficiency**: Deploying a model that can be integrated into a real-time clinical workflow.
- **Generalization**: Ensuring the model generalizes well to unseen data through robust data augmentation techniques.

# Project Workflow

<div align="center">
    <img width="500" alt="Screenshot 2024-10-14 at 1 48 36 PM" src="https://github.com/user-attachments/assets/a66917af-d4eb-4a6f-8e3d-593fbc065a67">
</div>
<br>

# Model Architecture

## Traditional U-Net Architecture

<div align="center">
<img width="500" alt="Screenshot 2024-10-14 at 2 01 36 PM" src="https://github.com/user-attachments/assets/619fe88a-d466-499e-a461-0045904ef106">
</div>
<br>

The U-Net architecture is a fully convolutional neural network primarily used for image segmentation. It was developed for biomedical image segmentation tasks and has gained widespread use due to its simplicity and effectiveness in accurately capturing both global and local context.

### Key components of U-Net:

- Encoder-Decoder Structure: U-Net consists of two main parts:
  1. **Encoder (Contracting Path)**: The left side of the U-Net is the encoder, which applies convolutional layers and downsampling (using max-pooling) to capture high-level features while reducing the spatial dimensions of the input.
  2. **Decoder (Expanding Path)**: The right side is the decoder, which applies upsampling layers to recover the original spatial dimensions and combines them with feature maps from the encoder through skip connections.
- **Skip Connections**: The skip connections help the decoder recover fine details by reintroducing feature maps from the corresponding encoder layers. This helps preserve spatial information that would otherwise be lost during downsampling.
  
### U-Net Workflow:
- The input image goes through a series of convolution and downsampling layers (encoder) to extract features.
- At the bottleneck, the most abstract features are captured.
- The decoder upsamples the feature maps and uses skip connections to combine low-level details from the encoder, producing a high-resolution segmentation mask.

# Attention U-Net Architecture

<div align="center"> <img width="500" alt="Screenshot 2024-10-04 at 10 28 53 PM" src="https://github.com/user-attachments/assets/5322a5c8-f2b1-4b5e-966d-669b66f8d29b"> </div> <br>

The Attention U-Net builds upon the traditional U-Net architecture by introducing attention gates into the decoder. These attention mechanisms allow the model to focus on the most relevant parts of the image, which improves segmentation accuracy by highlighting important regions and suppressing irrelevant background noise.

## Key components of Attention U-Net:

- **Attention Gates**: The attention gates refine the skip connections in the U-Net by allowing the model to selectively focus on relevant features from the encoder. This helps the model learn which parts of the image are most important for segmentation, improving performance, particularly for complex images where important regions may be small or hard to detect.

- **Skip Connections with Attention**: While U-Net uses direct skip connections, Attention U-Net adds attention gates to the feature maps before merging them in the decoder. This ensures that only the most relevant information is passed through to the decoder.

## Attention U-Net Workflow:
- **Encoder Path**: Similar to the traditional U-Net, the encoder path captures high-level features by applying a series of convolutional layers and downsampling.
- **Attention Gates**: In the decoder path, attention gates are applied to the feature maps from the encoder to focus on important regions, suppressing irrelevant information.
- **Decoder Path** : After passing through attention gates, the decoder upsamples the feature maps and combines them with the attended encoder features using concatenation, ultimately producing an output segmentation mask.

## Benefits of Attention U-Net:
- **Improved Segmentation**: By using attention gates, the model is better able to focus on relevant structures within the input image, leading to more accurate segmentation, especially in challenging regions.
- **Enhanced Skip Connections**: Attention gates refine the information passed from the encoder to the decoder, improving the effectiveness of the skip connections.

# Results
The Attention U-Net model performed exceptionally well on the brain tumor segmentation task, demonstrating strong generalization capabilities on both the validation and test datasets. The addition of attention gates helped focus on relevant regions in the MRI scans, improving segmentation accuracy.

## Performance Metrics:
- Test Accuracy: **97.72%**
- Test Dice Coefficient: **0.6848**
- Test Jaccard Index: **0.5207**
- Test Loss: **0.2109**

# Conclusion
The Attention U-Net architecture, with its focus on attention mechanisms, demonstrates significant improvements in segmentation accuracy for complex medical imaging tasks. By leveraging attention gates, the model is able to highlight the most important features, resulting in more precise segmentation of brain tumors. This project successfully illustrates how deep learning techniques can be applied to real-world medical challenges, providing a valuable tool for clinical applications.

## Future Work

The following project is currently under publication under the guidance of Dr.Prashant Kumar. The work highlights advancements in brain tumor segmentation using the Attention U-Net architecture and its potential applications in clinical settings.
