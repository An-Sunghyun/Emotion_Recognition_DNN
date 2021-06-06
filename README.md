# Emotion_Recognition_DNN
Tensorflow(VGG-16, Densenet-121, Densenet-201, Resnet-101), Pytorch(Resnet-18, Resnet-50, WideResnet-101, Densenet-201, EfficientNet-B5, SqueezeNet1.1)
Don't forget to modify the path setting since I didn't fix it.

## Tensorflow Result

### VGG-16

### Densenet-121

### Densenet-201

### Resnet-101

## Pytorch Result

### Densenet-201

### EfficientNet-B5

### SqueezeNet1.1

### Resnet-18

### Resnet-50

### WideResnet-101

## Data Set

I used commonly used datasets for the development of sentiment analyzers. The FER-2013 Faces Database. They are approximately 29000 image data, 48x48 size and classified into seven emotions. But One emotion, disgust, has not enough images so I decided to exclude it and train AI. This is because I heard that imbalance in datasets affects accuracy.
However, We go through the pre-process of resizing and arranging data for easy learning. Furthermore, image augmentation (zoom_range = 0.2, horizontal_flip=True, shear_range=0.2) is applied so that arbitrary data can be clearly classified.

## Face Detection


I used OpenCV library to recognize faces. I just added a little code to extract the face coordinates.

## Demonstration View in Python


The above photo shows the analysis result and photo printed in a Python environment in a softmax manner.

## Demonstration View in WebSite


The above photo shows the analysis result and photo printed in a website. User can upload image and can the result will come up right away just like image above.

## Reference Document


