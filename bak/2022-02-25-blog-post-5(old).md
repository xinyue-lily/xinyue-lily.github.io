---
layout: post
title: Blog Post 5
---


2022-02-25

# Image Classification

## Tensorflow

The Blog Post 5 is working in Google Colab. When training my model, enabling GPU lead to significant speed benefits.

## The Task
Classify cats and dogs

This is the link to my github respository. [https://github.com/xinyudong1129/Blog-Post/tree/main/blogpost5](https://github.com/xinyudong1129/Blog-Post/tree/main/blogpost5)

## Reference website
[Transfer leaning tutorials](https://www.tensorflow.org/tutorials/images/transfer_learning)

[Transfer leaning guide](https://tensorflow.google.cn/guide/keras/transfer_learning)

[Transfer learning & fine-tuning](https://keras.io/guides/transfer_learning/)

[Data augmentation](https://tensorflow.google.cn/tutorials/images/data_augmentation)

[Building powerful image classification models using very little data](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html)

## Load Packages and Obtain Data

Load Packages

```python

import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models,utils
from tensorflow.keras.preprocessing import image_dataset_from_directory

```

We'll use the dataset provided by the Tensorflow team that contains labeled images of cats and dogs.

```python
# location of data, code provided by professsor.
_URL = 'https://storage.googleapis.com/mledu-datasets/
        cats_and_dogs_filtered.zip'

# download the data and extract it
path_to_zip = utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)

# construct paths
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

# parameters for datasets
BATCH_SIZE = 32
IMG_SIZE = (160, 160)

# construct train and validation datasets 
train_dataset = utils.image_dataset_from_directory(train_dir,
                           shuffle=True,
                           batch_size=BATCH_SIZE,
                           image_size=IMG_SIZE)

validation_dataset = utils.image_dataset_from_directory(validation_dir,
                           shuffle=True,
                           batch_size=BATCH_SIZE,
                           image_size=IMG_SIZE)

# construct the test dataset by taking every 5th observation out of the validation dataset
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

```
We constructed three datasets: training_dataset, validation_dataset and test_dataset using image_dataset_from_directory.

```python
Downloading data from https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip
68608000/68606236 [==============================] - 1s 0us/step
68616192/68606236 [==============================] - 1s 0us/step
Found 2000 files belonging to 2 classes.
Found 1000 files belonging to 2 classes.
```

The following code is the technical code related to rapidly reading data provided by the professor.

```python

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

```
## 1.Visualiazation of images
We can get a piece of a dataset using the *take* method; will retrieve one batch(32 images with labels) from the training dataset.

We try to create a two-row visualizaiton. In the first row, show three random pictures of cats. In the second row, show three random pictures of dogs. Here's my code.

```python

cat = 0
dog = 0
plt.figure(figsize=(10, 6))
for images, labels in train_dataset.take(1):
  for i in range(32):
    if labels[i].numpy().astype("uint8")==0 and cat < 3:
      cat = cat + 1
      ax = plt.subplot(2, 3, cat)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title("cat")
     
    elif labels[i].numpy().astype("uint8")==1 and dog < 3:
      dog = dog + 1
      ax = plt.subplot(2, 3, dog+3)
      plt.imshow(images[i].numpy().astype("uint8"))
      plt.title("dog")
    plt.axis("off")
    if cat >= 3 and dog >= 3:
      break

```
Here's the resutls.

![PIC1.png](https://raw.githubusercontent.com/xinyue-lily/xinyue-lily.github.io/master/images/pic1_cat_dog.png)

## 2. First Model

We create a *tf.keras.Sequential* model named model1, which include two *Conv2D* layers, two *MaxPooling2D* layers, one *Flatten* layer, one *Dense* layer, one *Dropout* layer.

```python
model1 = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(160, 160, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.1)    
])
```

Train model1 and plot the history of the accuracy on both the training and vailidation sets.

```python
model1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model1.fit(train_dataset, 
           epochs=20, 
           validation_data=(validation_dataset))

```
Here's the history of the accuracy on both the training and vailidation sets.

```python
Epoch 1/20
63/63 [==============================] - 14s 61ms/step - loss: 16.4243 
          - accuracy: 0.5270 - val_loss: 0.8454 - val_accuracy: 0.5644
Epoch 2/20
63/63 [==============================] - 4s 55ms/step - loss: 1.1612 
        - accuracy: 0.6415 - val_loss: 0.7977 - val_accuracy: 0.6176
Epoch 3/20
63/63 [==============================] - 4s 55ms/step - loss: 1.0320 
        - accuracy: 0.7365 - val_loss: 1.1135 - val_accuracy: 0.5990
Epoch 4/20
63/63 [==============================] - 4s 54ms/step - loss: 0.8850 
        - accuracy: 0.7770 - val_loss: 1.0845 - val_accuracy: 0.6015
Epoch 5/20
63/63 [==============================] - 4s 54ms/step - loss: 0.6835 
        - accuracy: 0.8485 - val_loss: 1.3032 - val_accuracy: 0.6151
Epoch 6/20
63/63 [==============================] - 4s 58ms/step - loss: 0.6394 
        - accuracy: 0.8655 - val_loss: 1.8359 - val_accuracy: 0.5990
Epoch 7/20
63/63 [==============================] - 4s 57ms/step - loss: 0.5899 
        - accuracy: 0.8840 - val_loss: 1.6900 - val_accuracy: 0.6374
Epoch 8/20
63/63 [==============================] - 4s 54ms/step - loss: 0.4905 
        - accuracy: 0.8990 - val_loss: 1.9551 - val_accuracy: 0.6027
Epoch 9/20
63/63 [==============================] - 4s 57ms/step - loss: 0.5389 
        - accuracy: 0.8975 - val_loss: 1.8406 - val_accuracy: 0.6300
Epoch 10/20
63/63 [==============================] - 4s 58ms/step - loss: 0.4817 
        - accuracy: 0.9120 - val_loss: 1.9983 - val_accuracy: 0.6300
Epoch 11/20
63/63 [==============================] - 4s 60ms/step - loss: 0.5438 
        - accuracy: 0.8980 - val_loss: 1.9636 - val_accuracy: 0.6101
Epoch 12/20
63/63 [==============================] - 4s 56ms/step - loss: 0.5632 
        - accuracy: 0.9025 - val_loss: 2.2692 - val_accuracy: 0.6064
Epoch 13/20
63/63 [==============================] - 4s 59ms/step - loss: 0.6398 
        - accuracy: 0.8775 - val_loss: 2.0141 - val_accuracy: 0.6176
Epoch 14/20
63/63 [==============================] - 4s 59ms/step - loss: 0.4856 
        - accuracy: 0.9095 - val_loss: 2.3159 - val_accuracy: 0.6002
Epoch 15/20
63/63 [==============================] - 4s 58ms/step - loss: 0.4682 
        - accuracy: 0.9255 - val_loss: 2.6576 - val_accuracy: 0.6200
Epoch 16/20
63/63 [==============================] - 4s 60ms/step - loss: 0.5334 
        - accuracy: 0.9135 - val_loss: 2.8297 - val_accuracy: 0.6015
Epoch 17/20
63/63 [==============================] - 4s 60ms/step - loss: 0.4347 
        - accuracy: 0.9280 - val_loss: 2.7688 - val_accuracy: 0.6114
Epoch 18/20
63/63 [==============================] - 4s 60ms/step - loss: 0.4214 
        - accuracy: 0.9335 - val_loss: 2.8757 - val_accuracy: 0.6324
Epoch 19/20
63/63 [==============================] - 4s 59ms/step - loss: 0.4124 
        - accuracy: 0.9370 - val_loss: 3.0343 - val_accuracy: 0.6324
Epoch 20/20
63/63 [==============================] - 4s 59ms/step - loss: 0.4110 
        - accuracy: 0.9395 - val_loss: 3.0805 - val_accuracy: 0.6361
```

The validation accuracy of model1 stabilized between **56% to 64%** during training, better than the baseline.
plot the history of model1, including the training and validation performance.

```python
plt.plot(history.history["accuracy"], label = "training")
plt.plot(history.history["val_accuracy"], label = "validation")
plt.gca().set(xlabel = "epoch", ylabel = "accuracy")
plt.legend()
```
![PIC2.png](https://raw.githubusercontent.com/xinyue-lily/xinyue-lily.github.io/master/images/pic2_model1_history.png)

## 3.Model with Data Augmentation
In this part, we add some data augmentation layers to the model. Data Augmentation refers to include modified copies of the same image in the training set. For example, we can flip the image upside down or rotate it some degrees in order to help our model learn *invariant features* of input images.

we create a *tf.keras.layers.RandomFlip()* layer and a *tf.keras.layers.randomRotation()* layer. then plot both the origianl image and a few copies to which randomflip and randomrotation has been applied.

```python
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal")
])

for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
    plt.imshow(augmented_image[0]/ 255)
    plt.axis('off')

```

![PIC3.png](https://raw.githubusercontent.com/xinyue-lily/xinyue-lily.github.io/master/images/pic3_augmentation.png)

```python
data_rotation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.1),
])

for image, _ in train_dataset.take(1):
  plt.figure(figsize=(10, 10))
  first_image = image[0]
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    rotated_image = data_rotation(tf.expand_dims(first_image, 0))
    plt.imshow(rotated_image[0]/ 255)
    plt.axis('off')
```
![PIC4.png](https://raw.githubusercontent.com/xinyue-lily/xinyue-lily.github.io/master/images/pic4_rotation.png)

We create a new model called model2 in which the first two layers are augmentation layers, i.e. a data_augmentation layer and a data_rotation Layer.

```python
model2 = models.Sequential([
    data_augmentation,
    data_rotation,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.1)
])
```

```python
model2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model2.fit(train_dataset, 
           epochs=20, 
           validation_data=(validation_dataset)) 
```
Here's the history of the accuracy on both the training and vailidation sets of model2.

```python
Epoch 1/20
63/63 [==============================] - 7s 84ms/step - loss: 5.0798 
        - accuracy: 0.5010 - val_loss: 0.9185 - val_accuracy: 0.5173
Epoch 2/20
63/63 [==============================] - 5s 81ms/step - loss: 1.3856 
        - accuracy: 0.5595 - val_loss: 0.8037 - val_accuracy: 0.6572
Epoch 3/20
63/63 [==============================] - 5s 79ms/step - loss: 1.2619 
        - accuracy: 0.6025 - val_loss: 0.7720 - val_accuracy: 0.6287
Epoch 4/20
63/63 [==============================] - 5s 79ms/step - loss: 1.2941 
        - accuracy: 0.6040 - val_loss: 1.0175 - val_accuracy: 0.5743
Epoch 5/20
63/63 [==============================] - 5s 78ms/step - loss: 1.3219 
        - accuracy: 0.5905 - val_loss: 1.0052 - val_accuracy: 0.6176
Epoch 6/20
63/63 [==============================] - 5s 82ms/step - loss: 1.2011 
        - accuracy: 0.6145 - val_loss: 0.7295 - val_accuracy: 0.6522
Epoch 7/20
63/63 [==============================] - 5s 78ms/step - loss: 1.3301 
        - accuracy: 0.5825 - val_loss: 0.8652 - val_accuracy: 0.6089
Epoch 8/20
63/63 [==============================] - 5s 79ms/step - loss: 1.2173 
        - accuracy: 0.6380 - val_loss: 0.7083 - val_accuracy: 0.6683
Epoch 9/20
63/63 [==============================] - 5s 81ms/step - loss: 1.1707 
        - accuracy: 0.6475 - val_loss: 0.8769 - val_accuracy: 0.6584
Epoch 10/20
63/63 [==============================] - 5s 79ms/step - loss: 1.2206 
        - accuracy: 0.6395 - val_loss: 1.0213 - val_accuracy: 0.6931
Epoch 11/20
63/63 [==============================] - 5s 78ms/step - loss: 1.2456 
        - accuracy: 0.6445 - val_loss: 0.7617 - val_accuracy: 0.6844
Epoch 12/20
63/63 [==============================] - 5s 78ms/step - loss: 1.1505 
        - accuracy: 0.6715 - val_loss: 0.7246 - val_accuracy: 0.6696
Epoch 13/20
63/63 [==============================] - 5s 78ms/step - loss: 1.1822 
        - accuracy: 0.6715 - val_loss: 0.6808 - val_accuracy: 0.7054
Epoch 14/20
63/63 [==============================] - 5s 78ms/step - loss: 1.0823 
        - accuracy: 0.6870 - val_loss: 0.7188 - val_accuracy: 0.6832
Epoch 15/20
63/63 [==============================] - 5s 78ms/step - loss: 1.0803 
        - accuracy: 0.7025 - val_loss: 0.7774 - val_accuracy: 0.6696
Epoch 16/20
63/63 [==============================] - 5s 78ms/step - loss: 1.0588 
        - accuracy: 0.6955 - val_loss: 0.8687 - val_accuracy: 0.6819
Epoch 17/20
63/63 [==============================] - 5s 78ms/step - loss: 1.1105 
        - accuracy: 0.6895 - val_loss: 0.7198 - val_accuracy: 0.6881
Epoch 18/20
63/63 [==============================] - 5s 78ms/step - loss: 1.0650 
        - accuracy: 0.7075 - val_loss: 0.7951 - val_accuracy: 0.6696
Epoch 19/20
63/63 [==============================] - 5s 79ms/step - loss: 1.0901 
        - accuracy: 0.7015 - val_loss: 0.7840 - val_accuracy: 0.7005
Epoch 20/20
63/63 [==============================] - 5s 78ms/step - loss: 1.0995 
        - accuracy: 0.6915 - val_loss: 0.8102 - val_accuracy: 0.6795    
```

I tried some kinds of structures of the model, the training accuray of model2 is worse than model1, but validation accuray is **between 55% to 68%**, a little better than model1.
On the other hand, I observe overfitting in model2, in Epoch 13/20, the accuracy accuray is 70.54%, but from epoch 14 to 20, the training accuray decrease to 67.95%.

## 4.Data Processing
We can make simple transformations to the input data. The original data has pixels with RGB values between 0 adn 255, we can normalized the value between 0 to 1 or between -1 and 1 in order to train faster. The following code create a preprocessing layer called preprocessor and incorporate the lyaer as the very first layer of model3.

```python
i = tf.keras.Input(shape=(160,160,3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(i)
preprocessor = tf.keras.Model(inputs=[i],outputs=[x])

from tensorflow.keras import datasets, layers, models
model3 = models.Sequential([
    preprocessor,
    data_augmentation,
    data_rotation,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),    
])
```
Here's the history of the accuracy on both the training and vailidation sets of model3.

```python
model3.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model3.fit(train_dataset, 
           epochs=20, 
           validation_data=(validation_dataset)) 
```
```python
Epoch 1/20
63/63 [==============================] - 7s 88ms/step - loss: 0.8724 
        - accuracy: 0.5265 - val_loss: 0.6637 - val_accuracy: 0.5903
Epoch 2/20
63/63 [==============================] - 5s 76ms/step - loss: 0.6320 
        - accuracy: 0.6490 - val_loss: 0.6388 - val_accuracy: 0.5903
Epoch 3/20
63/63 [==============================] - 5s 76ms/step - loss: 0.6048 
        - accuracy: 0.6665 - val_loss: 0.5547 - val_accuracy: 0.7079
Epoch 4/20
63/63 [==============================] - 6s 85ms/step - loss: 0.5728 
        - accuracy: 0.7010 - val_loss: 0.5590 - val_accuracy: 0.7017
Epoch 5/20
63/63 [==============================] - 5s 76ms/step - loss: 0.5572 
        - accuracy: 0.7095 - val_loss: 0.6007 - val_accuracy: 0.6572
Epoch 6/20
63/63 [==============================] - 5s 76ms/step - loss: 0.5202 
        - accuracy: 0.7295 - val_loss: 0.5253 - val_accuracy: 0.7488
Epoch 7/20
63/63 [==============================] - 5s 77ms/step - loss: 0.5037 
        - accuracy: 0.7565 - val_loss: 0.5153 - val_accuracy: 0.7401
Epoch 8/20
63/63 [==============================] - 5s 77ms/step - loss: 0.4914 
        - accuracy: 0.7565 - val_loss: 0.4953 - val_accuracy: 0.7574
Epoch 9/20
63/63 [==============================] - 5s 77ms/step - loss: 0.4644 
        - accuracy: 0.7710 - val_loss: 0.4993 - val_accuracy: 0.7574
Epoch 10/20
63/63 [==============================] - 6s 82ms/step - loss: 0.4860 
        - accuracy: 0.7650 - val_loss: 0.5266 - val_accuracy: 0.7364
Epoch 11/20
63/63 [==============================] - 5s 76ms/step - loss: 0.4413 
        - accuracy: 0.7945 - val_loss: 0.5257 - val_accuracy: 0.7413
Epoch 12/20
63/63 [==============================] - 5s 76ms/step - loss: 0.4461 
        - accuracy: 0.7895 - val_loss: 0.5330 - val_accuracy: 0.7438
Epoch 13/20
63/63 [==============================] - 5s 76ms/step - loss: 0.4380 
        - accuracy: 0.7930 - val_loss: 0.5177 - val_accuracy: 0.7636
Epoch 14/20
63/63 [==============================] - 5s 77ms/step - loss: 0.4151 
        - accuracy: 0.8075 - val_loss: 0.4906 - val_accuracy: 0.7785
Epoch 15/20
63/63 [==============================] - 5s 76ms/step - loss: 0.3908 
        - accuracy: 0.8225 - val_loss: 0.5274 - val_accuracy: 0.7562
Epoch 16/20
63/63 [==============================] - 5s 75ms/step - loss: 0.4096 
        - accuracy: 0.8045 - val_loss: 0.4821 - val_accuracy: 0.7661
Epoch 17/20
63/63 [==============================] - 5s 76ms/step - loss: 0.3795 
        - accuracy: 0.8290 - val_loss: 0.5054 - val_accuracy: 0.7673
Epoch 18/20
63/63 [==============================] - 5s 77ms/step - loss: 0.3774 
        - accuracy: 0.8315 - val_loss: 0.4971 - val_accuracy: 0.7649
Epoch 19/20
63/63 [==============================] - 5s 78ms/step - loss: 0.3602 
        - accuracy: 0.8395 - val_loss: 0.4872 - val_accuracy: 0.7809
Epoch 20/20
63/63 [==============================] - 5s 77ms/step - loss: 0.3691 
        - accuracy: 0.8410 - val_loss: 0.6032 - val_accuracy: 0.7413
```
The validation accuray of model3 is **between 59% to 78%**, better than model1 and model2. I observe overfitting in model3 too, in Epoch 14/20, the accuracy accuray is 77.8%, but from epoch 15 to 18, the training accuray decreased. In epoth 19/20, the validation accuracy is 78.09%, but in epoth 20/20, the validation accuracy is only 74.13%.

## 5.Transfer Learning
Sometimes, someone might already have trained a model that does a related task, and might have learned some relevant patterns. We want to use a pre-exisiting model for our task.
Firstly, we access a pre-existing "base model", then we incorporate it into a full model for current task, and then train the model.

Here's our pre-exisiting "base model"
```python
# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                        include_top=False,
                        weights='imagenet')
base_model.trainable = False
i = tf.keras.Input(shape = IMG_SHAPE)
x = base_model(i, training = False)
base_model_layer = tf.keras.Model(inputs=[i], outputs = [x])
```
Create model4 that use MobileNetV2, the first layer is proprocessor, and then data augmentation layers, and then place base_model_layer, then globalMaxPooling2D, Dropout and dense layer at the very end to perform the classification.

```python
model4 = models.Sequential([
    preprocessor,
    data_augmentation,
    data_rotation,
    base_model_layer,
    layers.GlobalMaxPooling2D(),
    layers.Dropout(0.1),
    layers.Dense(1)
])
```
then train model4 and plot the history of model4.

```python
base_learning_rate = 0.0001
model4.compile(optimizer=tf.keras.optimizers.
                         Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
history = model4.fit(train_dataset, 
           epochs=20, 
           validation_data = (validation_dataset)) 
```
Here's the result.
```python
Epoch 1/20
63/63 [==============================] - 11s 114ms/step - loss: 1.1790 
          - accuracy: 0.6260 - val_loss: 0.5757 - val_accuracy: 0.7624
Epoch 2/20
63/63 [==============================] - 6s 84ms/step - loss: 0.6274 
        - accuracy: 0.7520 - val_loss: 0.3476 - val_accuracy: 0.8540
Epoch 3/20
63/63 [==============================] - 6s 86ms/step - loss: 0.4338 
        - accuracy: 0.8190 - val_loss: 0.2766 - val_accuracy: 0.8849
Epoch 4/20
63/63 [==============================] - 6s 84ms/step - loss: 0.3598 
        - accuracy: 0.8430 - val_loss: 0.2128 - val_accuracy: 0.9097
Epoch 5/20
63/63 [==============================] - 6s 84ms/step - loss: 0.3143 
        - accuracy: 0.8700 - val_loss: 0.1897 - val_accuracy: 0.9245
Epoch 6/20
63/63 [==============================] - 6s 84ms/step - loss: 0.2880 
        - accuracy: 0.8855 - val_loss: 0.1594 - val_accuracy: 0.9369
Epoch 7/20
63/63 [==============================] - 6s 84ms/step - loss: 0.2399 
        - accuracy: 0.9090 - val_loss: 0.1446 - val_accuracy: 0.9480
Epoch 8/20
63/63 [==============================] - 6s 96ms/step - loss: 0.2309 
        - accuracy: 0.9080 - val_loss: 0.1376 - val_accuracy: 0.9480
Epoch 9/20
63/63 [==============================] - 6s 84ms/step - loss: 0.2100 
        - accuracy: 0.9180 - val_loss: 0.1235 - val_accuracy: 0.9530
Epoch 10/20
63/63 [==============================] - 6s 84ms/step - loss: 0.2180 
        - accuracy: 0.9130 - val_loss: 0.1249 - val_accuracy: 0.9542
Epoch 11/20
63/63 [==============================] - 6s 83ms/step - loss: 0.2146 
        - accuracy: 0.9250 - val_loss: 0.1223 - val_accuracy: 0.9517
Epoch 12/20
63/63 [==============================] - 6s 85ms/step - loss: 0.1791 
        - accuracy: 0.9350 - val_loss: 0.1031 - val_accuracy: 0.9604
Epoch 13/20
63/63 [==============================] - 6s 84ms/step - loss: 0.1902 
        - accuracy: 0.9170 - val_loss: 0.1134 - val_accuracy: 0.9554
Epoch 14/20
63/63 [==============================] - 5s 83ms/step - loss: 0.1524 
        - accuracy: 0.9440 - val_loss: 0.1025 - val_accuracy: 0.9567
Epoch 15/20
63/63 [==============================] - 6s 84ms/step - loss: 0.1690 
        - accuracy: 0.9335 - val_loss: 0.1046 - val_accuracy: 0.9592
Epoch 16/20
63/63 [==============================] - 6s 84ms/step - loss: 0.1795 
        - accuracy: 0.9375 - val_loss: 0.1015 - val_accuracy: 0.9592
Epoch 17/20
63/63 [==============================] - 5s 83ms/step - loss: 0.1553 
        - accuracy: 0.9395 - val_loss: 0.0880 - val_accuracy: 0.9653
Epoch 18/20
63/63 [==============================] - 6s 85ms/step - loss: 0.1488 
        - accuracy: 0.9420 - val_loss: 0.1000 - val_accuracy: 0.9616
Epoch 19/20
63/63 [==============================] - 6s 83ms/step - loss: 0.1336 
        - accuracy: 0.9465 - val_loss: 0.0935 - val_accuracy: 0.9616
Epoch 20/20
63/63 [==============================] - 6s 86ms/step - loss: 0.1303 
        - accuracy: 0.9445 - val_loss: 0.0829 - val_accuracy: 0.9641
```
The validation accuray of model4 is **between 77% to 97%**, better than model1, model2 and model3.


Score the Test Dataset, The test accuracy is 97.4%.

```python
initial_epochs = 10
loss,accuray = model5.evaluate(test_dataset)
accuray
```
```python
6/6 [==============================] - 1s 64ms/step - loss: 0.0541 - accuracy: 0.9740
[0.05411681905388832, 0.9739583134651184]
```

Let's take a look at the learning curve of training and verifying accuracy / loss when using mobilenet V2 basic model as a fixed feature extraction program.

```python
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
```
![PIC5.png](https://raw.githubusercontent.com/xinyue-lily/xinyue-lily.github.io/master/images/pic5_model4.png)

Thank you!!!