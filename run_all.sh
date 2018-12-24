#!/bin/bash

models=("ResNet50" "VGG19" "AlexNet" "GoogleNet" "InceptionV4" "MobileNet" "MobileNetV2" "DistResNet" "SE_ResNeXt50_32x4d")

for model in  ${models[@]}
do
    echo $mdoel
    python train.py --model=$model
done
