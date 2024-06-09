# Brain MRI Image Classification using Convolutional Neural Network

Project Description
This project aims to classify brain MRI images using a Convolutional Neural Network (CNN). The primary goal is to detect the presence or absence of specific features in brain MRI images. The model leverages CNN architecture to effectively recognize patterns in image data.

Data - https://www.kaggle.com/datasets/luluw8071/brain-tumor-mri-datasets

The dataset contains brain MRI images divided into two classes:

"yes": images containing certain features.
"no": images not containing those features.

Training data constitutes 80% of the dataset, while testing data constitutes 20%.

MRI Image Examples

![image](https://github.com/TanskiSzymon/ML/assets/108231030/7abc72cf-4e35-42c9-b398-023b5bf40991)

Figure 1. MRI images classified as: 1 – yes and 2 – no.

Model Architecture
The model consists of two main blocks:

Convolutional Layers: Two convolutional layers, each followed by ReLU activation and MaxPooling layers.
Fully Connected Layers: Two fully connected layers with ReLU activation and LogSoftmax output.
Results
The model was trained for 10 epochs using the Adam optimizer and early stopping at epoch 8. The model achieved an accuracy of 99.67% on the training set and 98.80% on the test set.

# Plots:

![image](https://github.com/TanskiSzymon/ML/assets/108231030/312e5296-63fd-46a4-8258-9b8903fed8a5)

Training Loss over Epochs

![image](https://github.com/TanskiSzymon/ML/assets/108231030/3f92f449-7331-4b0e-901b-4f3f006ce3c4)

Training Accuracy over Epochs

![image](https://github.com/TanskiSzymon/ML/assets/108231030/6cbc6d6a-6443-46d5-b3c8-140c069514ff)

Confusion Matrix



# Conclusions

The model achieved high accuracy on both the training and test sets, indicating that it effectively learned to classify brain MRI images. The training process was significantly accelerated using Apple's Metal Performance Shaders (MPS), reducing the training time from 30 hours to just 2 minutes.

The model shows good performance, as confirmed by the confusion matrix results, which indicate high precision, recall, and specificity. The worst case is a False Negative (FN), as it means a patient might not receive timely help. However, the number of such errors is low (6 cases), and achieving 100% certainty is not feasible.

If you have any questions or suggestions about the project, please feel free to contact me.
