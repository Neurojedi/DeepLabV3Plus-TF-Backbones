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
    'efficientnetb0': {'model': EfficientNetB0, 'feature_1': 'block7a_expand_activation', 'feature_2': 'block3a_expand_activation'},
    'efficientnetb1': {'model': EfficientNetB1,'feature_1': 'block7a_expand_activation', 'feature_2': 'block3a_expand_activation'},
    'efficientnetb2': {'model': EfficientNetB2, 'feature_1': 'block7a_expand_activation','feature_2': 'block3a_expand_activation'},
    'efficientnetb3': {'model': EfficientNetB3,'feature_1': 'block7a_expand_activation','feature_2': 'block3a_expand_activation'},
    'efficientnetb4': {'model': EfficientNetB4,'feature_1': 'block7a_expand_activation','feature_2': 'block3a_expand_activation'},
    'efficientnetb5': {'model': EfficientNetB5,'feature_1': 'block6a_expand_activation','feature_2': 'block3a_expand_activation'},
    'efficientnetb6': {'model': EfficientNetB6, 'feature_1': 'block6a_expand_activation','feature_2': 'block3a_expand_activation'},
    'efficientnetb7': {'model': EfficientNetB7, 'feature_1': 'block6a_expand_activation','feature_2': 'block3a_expand_activation'},
    'efficientnetv2b0': {'model': EfficientNetV2B0,'feature_1': 'block6g_activation','feature_2': 'block3a_expand_activation'},
    'efficientnetv2b1': {'model': EfficientNetV2B1,'feature_1': 'block6g_activation','feature_2': 'block3a_expand_activation'},
    'efficientnetv2b2': {'model': EfficientNetV2B2, 'feature_1': 'block6g_activation','feature_2': 'block3a_expand_activation'},
    'efficientnetv2b3': {'model': EfficientNetV2B3, 'feature_1': 'block6g_activation','feature_2': 'block3a_expand_activation'},
    'mobilenet': { 'model': MobileNet, 'feature_1': 'conv_pw_11_relu','feature_2': 'conv_pw_3_relu'},
    'mobilenetv2': { 'model': MobileNetV2, 'feature_1': 'block_15_expand_relu','feature_2': 'block_2_expand_relu'},
    'vgg16': { 'model': VGG16, 'feature_1': 'block5_conv3', 'feature_2': 'block3_conv3'},
    'vgg19': { 'model': VGG19, 'feature_1': 'block5_conv4', 'feature_2': 'block3_conv4'},
    'inceptionv3': {'model': InceptionV3,'feature_1': 'activation_182', 'feature_2': 'activation_100'},
    'nasnetlarge': {'model': NASNetLarge,'feature_1': 'separable_conv_1_normal_left1_14', 'feature_2': 'separable_conv_1_reduction_right2_stem_2'},
    'convnexttiny': {'model': ConvNeXtTiny, 'feature_1': 'convnext_tiny_stage_3_block_0_depthwise_conv', 'feature_2': 'convnext_tiny_stage_1_block_0_depthwise_conv'},
    'convnextbase': {'model': ConvNeXtBase, 'feature_1': 'convnext_tiny_stage_3_block_0_depthwise_conv', 'feature_2': 'convnext_base_stage_1_block_1_depthwise_conv'},
    'convnextsmall': {'model': ConvNeXtSmall,'feature_1': 'convnext_small_stage_2_block_19_depthwise_conv', 'feature_2': 'convnext_small_stage_1_block_2_depthwise_conv'},
    'convnextlarge': {'model': ConvNeXtLarge,'feature_1': 'convnext_large_stage_2_block_17_depthwise_conv', 'feature_2': 'convnext_large_stage_1_block_2_depthwise_conv'}
}
