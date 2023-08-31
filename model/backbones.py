import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.layers import AveragePooling2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras.applications import (
    ResNet50, ResNet101, DenseNet121, DenseNet169, ResNet50V2, ResNet101V2,
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
    EfficientNetV2B0,EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3, MobileNet,
    MobileNetV2, VGG16, VGG19, InceptionV3, ConvNeXtSmall, ConvNeXtTiny, ConvNeXtBase,
    ConvNeXtLarge, NASNetLarge
)

# Experimental Backbones
backbones_list = {
    'resnet50': {'model': ResNet50, 'feature_1': 'conv4_block6_2_relu', 'feature_2': 'conv2_block3_2_relu'},
    'resnet101': {'model': ResNet101,'feature_1': 'conv4_block19_2_relu', 'feature_2': 'conv2_block3_2_relu'},
    'resnet50v2': {'model': ResNet50V2, 'feature_1': 'conv4_block4_1_relu', 'feature_2': 'conv2_block3_1_relu'},
    'resnet101v2': {'model': ResNet101V2, 'feature_1': 'conv4_block23_1_relu', 'feature_2': 'conv2_block3_1_relu'},
    'densenet121': {'model': DenseNet121, 'feature_1': 'relu', 'feature_2': 'conv2_block6_0_relu'},
    'densenet169': {'model': DenseNet169, 'feature_1': 'relu','feature_2': 'conv3_block5_1_relu'},
    'mobilenet': { 'model': MobileNet, 'feature_1': 'conv_pw_11_relu','feature_2': 'conv_pw_3_relu'},
    'mobilenetv2': { 'model': MobileNetV2, 'feature_1': 'block_15_expand_relu','feature_2': 'block_2_expand_relu'},
    'vgg16': { 'model': VGG16, 'feature_1': 'block5_conv3', 'feature_2': 'block3_conv3'},
    'vgg19': { 'model': VGG19, 'feature_1': 'block5_conv4', 'feature_2': 'block3_conv4'}
}
