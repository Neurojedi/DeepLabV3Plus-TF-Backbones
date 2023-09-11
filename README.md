# DeepLabV3Plus-TF-Backbones (Under Construction 🛠️)

This repository extends the Keras example code for [Multiclass semantic segmentation using DeepLabV3+](https://keras.io/examples/vision/deeplabv3_plus/). While the original example utilized a ResNet50 backbone, my work focuses on adapting the network to support various backbones available in `tensorflow.keras.applications`.


Currently, the model can be used with the following backbones:

1. `ResNet50`
2. `ResNet101`
3. `ResNet50V2`
4. `ResNet101V2`
5. `DenseNet121`
6. `DenseNet169`
7. `MobileNet`
8. `MobileNetV2`
9. `VGG16`
10. `VGG19`

In my experiments, I found the following backbones were ineffective:

1. `ConvNeXtSmall`
2. `ConvNeXtTiny`
3. `ConvNeXtBase`
4. `ConvNeXtLarge`
5. `EfficientNetB0`
6. `EfficientNetB1`
7. `EfficientNetB2`
8. `EfficientNetB3`
9. `EfficientNetB4`
10. `EfficientNetB5`
11. `EfficientNetB6`
12. `EfficientNetB7`
