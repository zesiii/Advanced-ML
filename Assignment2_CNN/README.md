# Explore CNN techniques in Medical Image Segmentation
## Contents
- [Overview](#overview)
- [Steps](#steps)
- [Results](#results)
- [Conclusions](#Conclusion)
- [Insights](#Insights)


## Overview
This project explores various CNN structures and transfer models in medical image classification. The dataset used for this project is provided by: M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, “Can AI help in screening Viral and COVID-19 pneumonia?” arXiv preprint, 29 March 2020, [Link](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Farxiv.org%2Fabs%2F2003.13145)

We used 1344 photos from each of the 3 categories: Covid, Normal and Viral Pneumonia, so the dataset size is 4302. 

Because of the limited dataset size, we performed data augmentation on the training set through randomly rescaling, rotating, resizing, etc., and got a much larger training set that can yield more robust neural network. 

2 top scoring CNN structures are built by us from scratch and the 3rd performing one is a transfer model based on VGG16. 

We then tried out VGG-UNet, a transfer model specialized in medical image segmentation. This model displayed high convergence rate. 

[Front-end Notebook](assignment2.ipynb) is the notebook where you can use our modularized code on the CNN2 model we built to predict a new set of images.It contains preprocessing, augmentation and load weights to predict. [Modules](assignment2.ipynb) is the back-end code to perform these functions.

[Report](assignment2_2_yz4691_qg2218.ipynb) is the detailed report on our data processing and explorations. 

## Steps
### Exlore the dataset
#### Image of the 3 Categries
![image](https://github.com/zesiii/Advanced-ML/assets/144854988/75e1a3f6-319c-438e-9d50-f51ca5601530)
#### Edge Detection
![image](https://github.com/zesiii/Advanced-ML/assets/144854988/efd1a5f8-d57e-43ef-991a-93f7532c1200)

### Data Augmentation
Due to the limited dataset size of medical images, data augmentation is often performed in such tasks and similarly in our project. We used ImageDataGenerator from keras to rescale, rotate, shift, flip and fill the training set we are provided.

### Preprocess
1. convert to RGB
2. min max transformation to normalize
3. resize to (192,192,3)

### Training 2 CNN models from scratch
The first CNN model has basic blocks of Conv2D + Batchnormalization + Conv2D + MaxPooling2D. We added dropout layer after some blocks to prevent overfitting. 
The second CNN model uses Conv2D + MaxPooling2D as basic structure. Dropout is added before output layer to prevent overfitting. 
|Loss Function|Optimizer|
|:-:|:-:|
|`Categorical Cross Entropy`|`Adam`|

### Tuning 3 Transfer Models (EfficientNetB3, ResNet50, VGG16)
1. Define the transfer model
2. freeze the bottom layers pre-trained model, leaving top 5 layers trainable (the exact number depends on every model and dataset)
|Loss Function|Optimizer|
|:-:|:-:|
|`Categorical Cross Entropy`|`Adam`|

## Results
|Model|Accuracy|Training History|
|:-:|:-:|:-:|
|CNN1|79%|![image](https://github.com/zesiii/Advanced-ML/assets/144854988/d7aa3ff9-4fe7-41f2-872a-b344cc8fdc03)|
|CNN2|79%|![image](https://github.com/zesiii/Advanced-ML/assets/144854988/9d8faf38-fa12-46b4-85e5-1c8ff7a38035)|
|ResNet50|58%|![image](https://github.com/zesiii/Advanced-ML/assets/144854988/f0b1dc63-6aac-45b7-aace-a11ea03d3fea)|
|VGG-16|63%|![image](https://github.com/zesiii/Advanced-ML/assets/144854988/01204622-0adc-406b-99e2-b96654836b78)|
|VGG-UNet|64%|![image](https://github.com/zesiii/Advanced-ML/assets/144854988/59c69a23-6df4-44e1-9953-da0f6a63db3b)|

## Conclusions
1. We performed data augmentation on giving image set and preprocessed it to fit the input size of CNN. We then splitted the augmented dataset into train and validation set.
2. We performed edge detection before diving into CNN to get some preliminary insights on the data. The result showed some degree of pattern within each outcome category.
3. We built 2 CNN models from scratch, each has achieved accuracy about 80% on validation set.
4. We built 3 transfer learning models based on EfficientNetB3, ResNet50 and VGG16. ResNet50 converges the fastest and VGG16 has best performance of the 3.
5. We built another transfer model based on VGG-Unet, which converges very fast and has good performance.
   
## Insights
1. Data augmentation can greatly increase training set size of images, but at the cost of high demand of GPU resource and computational power. 
2. When building CNN from scratch, we can evaluate preliminary model and adjust its structure based on performance metric. For example, if the model is converging too slow, try deleting some dropout layers or decrease the rate.
3. When choosing transfer model, consider the relative complexity of the model to the task.


