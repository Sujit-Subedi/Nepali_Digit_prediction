# Nepali Digit Detection using CNN

This repository contains a Convolutional Neural Network (CNN) pipeline implemented in PyTorch for detecting Nepali digits from images. 

## Code Flow

### 1. Dataloader.py

This script defines a class `Dataloading` for loading and preprocessing the dataset. It loads images from the specified directory and creates data loaders for training and testing.

### 2. CNN_model.py

Defines the architecture of the CNN model using PyTorch's `nn.Module`. The model consists of convolutional layers, activation functions, pooling layers, and fully connected layers.

### 3. Model_training.py

Contains the logic for training the CNN model. It uses the dataset loaders created using `Dataloader.py`. The training process includes forward pass, backward pass, optimization, and evaluation.

### 4. inference.py

Provides functions for loading a trained model and performing inference on new images. It loads a trained model, preprocesses the input image, and generates predictions.

## CNN Architecture Summary

| Layer            | Description                                                                                   |
|------------------|-----------------------------------------------------------------------------------------------|
| Conv1            | Input Channels: 1 <br>  Output Channels: 32 <br> Kernel Size: (3, 3) <br> Stride: 1 <br>  Padding: 1 <br>  Activation: ReLU             |
| Conv2            |  Input Channels: 32 <br>  Output Channels: 32 <br> Kernel Size: (3, 3) <br>  Stride: 1 <br>  Padding: 1 <br>  Activation: ReLU             |
| Max Pooling      |  Kernel Size: (2, 2)                                                                         |
| FC3              |  Input Size: 8192 (flattened feature map) <br> Output Size: 512 <br>  Activation: ReLU   |
| FC4              |  Input Size: 512 <br>  Output Size: 10 (number of classes)                                  |
| Activation Funcs |  ReLU (Used after each convolutional layer for non-linearity)                                |
| Dropout          |  Used after the first convolutional layer                                                    |

## Training Outline

On training the 17000+ B/W 32*32 image of 10 different each representing Nepali digit for 0-9 based on following hyperparmeter. Such loss and accuracy is obtained.

### Experiment 1
| Parameter        | Value                    |
|------------------|--------------------------|
| Loss Function    | nn.CrossEntropyLoss()    |
| Learning Rate    | 0.001                    |
| Momentum         | 0.9                      |
| Optimizer        | "SGD"                    |
| Number of Epochs | 20                       |
| Batch Size       | 64                       |
| Activation fun   | ReLU                     |
| Kernel   | 3,3                    |


#### Graphs and Visualization:

![Example image](Graphs/Img.png)

### Experiment 2
| Parameter        | Value                    |
|------------------|--------------------------|
| Loss Function    | nn.CrossEntropyLoss()    |
| Learning Rate    | 0.001                    |
| Momentum         | 0.9                      |
| Optimizer        | "Adam"                   |
| Number of Epochs | 15                        
| Batch Size       | 64                       |
| Activation fun   | GeLU                     |
| Kernel   | 2,2                    |


#### Graphs and Visualization:

![Example image](Graphs/2,2_kernel_GELU_Adam.png)

## Conclusion

- In Experimentaion 1 and 2 Train accuracy is in the range of 97-98% but the difference between validation is different.

- **Reason behind large Gap in Experiment 2**

    - Memorization: The model might be memorizing the specific patterns in the training data rather than learning general features that can be applied to unseen data.

    - Poor Generalization: While the model performs well on the training data, it may not be able to accurately classify new examples it encounters.

## Reference

[Pytorch Documentation.](https://pytorch.org/docs/stable/index.html)

[How Do Convolutional Layers Work in Deep Learning Neural Networks? - MAchine learning mastery](https://machinelearningmastery.com/convolutional-layers-for-deep-learning-neural-networks/)





