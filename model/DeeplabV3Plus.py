import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from backbones import backbones_list as BACKBONES
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D
from tensorflow.keras.applications import (
    ResNet50, ResNet101, DenseNet121, DenseNet169, ResNet50V2, ResNet101V2,
    EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3,
    EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
    EfficientNetV2B0,EfficientNetV2B1, EfficientNetV2B2, EfficientNetV2B3,
    MobileNetV2, VGG16, VGG19, InceptionV3, ConvNeXtSmall, ConvNeXtTiny, ConvNeXtBase,
    ConvNeXtLarge, NASNetLarge
)

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = layers.Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=keras.initializers.HeNormal(),
    )(block_input)
    x = layers.BatchNormalization()(x)
    return tf.nn.relu(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = layers.AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output


def DeeplabV3Plus(image_size, num_classes, weights="imagenet", backbone="resnet50", output_act="sigmoid"):
    model_input = keras.Input(shape=(image_size, image_size, 3))
    if backbone in BACKBONES:
        base_model = BACKBONES[backbone]['model'](weights='imagenet', include_top=False, input_tensor=model_input)
        feature_1 = BACKBONES[backbone]['feature_1']
        feature_2 = BACKBONES[backbone]['feature_2']
    else:
      return print("Backbone is not found. Current avaliable backbones: ")  
        
    x = base_model.get_layer(feature_1).output
    x = DilatedSpatialPyramidPooling(x)

    input_a = layers.UpSampling2D(
        size=(image_size // 4 // x.shape[1], image_size // 4 // x.shape[2]),
        interpolation="bilinear",
    )(x)
    input_b = base_model.get_layer(feature_2).output
    input_b = layers.UpSampling2D(
        size=(image_size // 4 // input_b.shape[1], image_size // 4 // input_b.shape[2]),
        interpolation="bilinear",
    )(input_b)
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = layers.Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = layers.UpSampling2D(
        size=(image_size // x.shape[1], image_size // x.shape[2]),
        interpolation="bilinear",
    )(x)
    model_output = layers.Conv2D(num_classes, kernel_size=(1, 1), padding="same")(x)
    model_output = Activation(output_act)(model_output)

    return keras.Model(inputs=model_input, outputs=model_output)



