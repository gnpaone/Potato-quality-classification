# Potato-Quality-Classification

## Table of Contents
+ [About](#about)
+ [Exploratory Data Analysise](#exploratory-data-analysis)
+ [Model](#model)
+ [Training](#training)
+ [Results Evaluation](#results-evaluation)
+ [Conclusion](#conclusion)
+ [Reference](#reference)

## About

Potato Quality dataset is a labelled data set with 2 different quality classes. Each class contains 1000s of images. Using the data provided, a deep learning model built on TensorFlow is trained to classify into various classes in dataset.

**Classes:** (Defective, Non-Defective)
<br>**Epoches:** 50
<br>**Batch_size:** 32

Images are split to train and test set with 769 images belonging to 2 different classes. 

## Exploratory Data Analysis

Let's preview some of the images.

<img src = "https://github.com/gnpaone/Potato-quality-classification/blob/main/Pictures/EDA.png">

The size of the images are mostly consistent, so all the images are scaled to same size, so we dont have to worry about inconsistency.

## Model
To create a convolution neural network to classfied the images, Sequential model is used.

```python
model = models.Sequential([
    resize_and_rescale,
    data_augmentation,
    layers.Conv2D(32,(3,3),activation = 'relu',input_shape = input_shape),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64,(3,3),activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128,(3,3),activation = 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(256,activation = 'relu'),
    layers.Dense(n_classes,activation = 'softmax'),
])
```

## Training

<img src = "https://github.com/gnpaone/Potato-quality-classification/blob/main/Pictures/training.png">

Model accuracy increased over each epoch, overfitting started at around 3 epochs. The model achieved validation accuracy of **99.34%** with a 0.0397 cross entropy validation loss.

## Results Evaluation

Preview some predictions from the model:

<p>First Image to Predict:</p>
<img src = "https://github.com/gnpaone/Potato-quality-classification/blob/main/Pictures/test.png">
<p>Actual Label: imarti</p>
<p>Predicted Label: imarti</p>

Now, let's examine in more detail how the model performs and evaluate those predictions.

<img src = "https://github.com/gnpaone/Potato-quality-classification/blob/main/Pictures/model.png">

## Conclusion

With the given data sets for 2 classes of quality: Defective and Non-Defective, the model final accuracy reached 99.34% with cross entropy validation loss of 0.0397.

## Reference

[Potato Quality Dataset](Dataset/Pepsico-RnD-Potato-Lab-Dataset)
- [PepsiCo Lab Potato Chips Quality Control](https://www.kaggle.com/datasets/concaption/pepsico-lab-potato-quality-control)

<p align="center">
<b>⭐ Please consider starring this repository if it helped you! ⭐</b>
</p>
