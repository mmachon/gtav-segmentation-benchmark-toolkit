# Keras implementation of SafeUAV-Nets
# Source to original code (pytorch):
#   https://gitlab.com/mihaicristianpirvu/SafeUAV
#   https://sites.google.com/site/aerialimageunderstanding/safeuav-learning-to-estimate-depth-and-safe-landing-areas-for-uavs
#   Marcu, Alina and Costea, Dragos and Licaret, Vlad and Pirvu, Mihai and Leordeanu, Marius and Slusanschi, Emil.
#   "SafeUAV: Learning to estimate depth and safe landing areas for UAVs from synthetic data."
#   European Conference on Computer Vision (ECCV) UAVision Workshop. 2018.


from tensorflow.keras import layers, optimizers, losses, models

from .model_backend import ModelBackend


class SaveUAVBackend(ModelBackend):

    def __init__(self, backbone, chip_size):
        super().__init__(chip_size)
        self.backbone = backbone

    def compile(self):
        if self.backbone == "large":
            model_backend = self.build_saveuav_large()
        elif self.backbone == "small":
            model_backend = self.build_saveuav_small()
        else:
            return None
        model_backend.compile(
            optimizer=optimizers.Adam(lr=1e-4),
            loss=losses.CategoricalCrossentropy(),
            metrics=self.metrics
        )
        return model_backend

    def build_bottleneck_block(self, num_filters, x):
        bottleneck_layers = []
        for i in range(6):

            bottleneck_layers.append(layers.Conv2D(filters=num_filters * 8, kernel_size=3, dilation_rate=2**i, padding="same",
                          activation="relu")(x if len(bottleneck_layers) == 0 else bottleneck_layers[-1]))
        x = layers.Concatenate()(bottleneck_layers)
        return x

    def build_saveuav_large(self):
        num_filters = 64
        size = self.chip_size
        input = layers.Input((size, size, 3))
        down1 = layers.Conv2D(filters=num_filters, kernel_size=3, activation="relu", padding="same")(input)
        down1 = layers.Conv2D(filters=num_filters, kernel_size=3, activation="relu", padding="same")(down1)
        down1pool = layers.MaxPool2D(padding='same')(down1)

        down2 = layers.Conv2D(filters=num_filters * 2, kernel_size=3, activation="relu", padding="same")(down1pool)
        down2 = layers.Conv2D(filters=num_filters, kernel_size=3, activation="relu", padding="same")(down2)
        down2pool = layers.MaxPool2D(padding='same')(down2)

        down3 = layers.Conv2D(filters=num_filters * 4, kernel_size=3, activation="relu", padding="same")(down2pool)
        down3 = layers.Conv2D(filters=num_filters, kernel_size=3, activation="relu", padding="same")(down3)
        down3pool = layers.MaxPool2D(padding='same')(down3)

        y_dilate_concat = self.build_bottleneck_block(num_filters, down3pool)

        up3_tr = layers.Conv2DTranspose(filters=num_filters * 4, kernel_size=3, strides=(2, 2), activation="relu",
                                        padding="same")(y_dilate_concat)
        y_up3 = layers.Concatenate()([up3_tr, down3])
        y_up3 = layers.Conv2D(filters=num_filters * 4, kernel_size=3, activation="relu", padding="same")(y_up3)
        y_up3 = layers.Conv2D(filters=num_filters * 4, kernel_size=3, activation="relu", padding="same")(y_up3)
        y_up3 = layers.Conv2D(filters=num_filters*4, kernel_size=3, activation="relu", padding="same")(
            y_up3)

        up2_tr = layers.Conv2DTranspose(filters=num_filters * 2, kernel_size=3, strides=(2, 2), activation="relu",
                                        padding="same")(y_up3)
        y_up2_tr_add = layers.Concatenate()([up2_tr, down2])
        y_up2 = layers.Conv2D(filters=num_filters * 2, kernel_size=3, activation="relu", padding="same")(y_up2_tr_add)
        y_up2 = layers.Conv2D(filters=num_filters * 2, kernel_size=3, activation="relu", padding="same")(y_up2)
        y_up2_tr_add = layers.Conv2D(filters=num_filters*2, kernel_size=3, activation="relu", padding="same")(
            y_up2)

        up1_tr = layers.Conv2DTranspose(filters=num_filters, kernel_size=3, strides=(2, 2), activation="relu",
                                        padding="same")(y_up2_tr_add)
        y_up1_tr_add = layers.Concatenate()([up1_tr, down1])
        y_up1 = layers.Conv2D(filters=num_filters, kernel_size=3, activation="relu", padding="same")(y_up1_tr_add)
        y_up1 = layers.Conv2D(filters=num_filters, kernel_size=3, activation="relu", padding="same")(y_up1)
        y_up1_tr_add = layers.Conv2D(filters=num_filters, kernel_size=3, activation="relu", padding="same")(y_up1)

        final_conv = layers.Conv2D(filters=6, kernel_size=1)(y_up1_tr_add)
        x = layers.Activation("softmax")(final_conv)

        model = models.Model(input, x)

        return model

    def build_saveuav_small(self):
        num_filters = 16
        size = self.chip_size
        input = layers.Input((size, size, 3))
        down1 = layers.Conv2D(filters=num_filters, kernel_size=3, activation="relu", padding="same")(input)
        down1pool = layers.Conv2D(filters=num_filters, kernel_size=3, strides=(2, 2), activation="relu", padding="same")(down1)
        down2 = layers.Conv2D(filters=num_filters*2, kernel_size=3, activation="relu", padding="same")(down1pool)
        down2pool = layers.Conv2D(filters=num_filters*2, kernel_size=3, strides=(2, 2), activation="relu", padding="same")(down2)
        down3 = layers.Conv2D(filters=num_filters*4, kernel_size=3, activation="relu", padding="same")(down2pool)
        down3pool = layers.Conv2D(filters=num_filters*4, kernel_size=3, strides=(2, 2), activation="relu", padding="same")(down3)

        y_dilate1 = layers.Conv2D(filters=num_filters * 8, kernel_size=3, dilation_rate=1, padding="same",
                                  activation="relu")(down3pool)
        y_dilate2 = layers.Conv2D(filters=num_filters * 8, kernel_size=3, dilation_rate=2, padding="same",
                                  activation="relu")(y_dilate1)
        y_dilate3 = layers.Conv2D(filters=num_filters * 8, kernel_size=3, dilation_rate=4, padding="same",
                                  activation="relu")(y_dilate2)
        y_dilate4 = layers.Conv2D(filters=num_filters * 8, kernel_size=3, dilation_rate=8, padding="same",
                                  activation="relu")(y_dilate3)
        y_dilate5 = layers.Conv2D(filters=num_filters * 8, kernel_size=3, dilation_rate=16, padding="same",
                                  activation="relu")(y_dilate4)
        y_dilate6 = layers.Conv2D(filters=num_filters * 8, kernel_size=3, dilation_rate=32, padding="same",
                                  activation="relu")(y_dilate5)
        y_dilate_sum = layers.Add()([y_dilate1, y_dilate2, y_dilate3, y_dilate4, y_dilate5, y_dilate6])

        up3_tr = layers.Conv2DTranspose(filters=num_filters*4, kernel_size=3, strides=(2, 2), activation="relu", padding="same")(y_dilate_sum)
        y_up3_tr_add = layers.Add()([up3_tr, down3])
        y_up3 = layers.Conv2D(filters=num_filters*4, kernel_size=3, activation="relu", padding="same")(y_up3_tr_add)

        up2_tr = layers.Conv2DTranspose(filters=num_filters*2, kernel_size=3, strides=(2, 2), activation="relu", padding="same")(y_up3)
        y_up2_tr_add = layers.Add()([up2_tr, down2])
        y_up2 = layers.Conv2D(filters=num_filters*2, kernel_size=3, activation="relu", padding="same")(y_up2_tr_add)

        up1_tr = layers.Conv2DTranspose(filters=num_filters, kernel_size=3, strides=(2, 2), activation="relu", padding="same")(y_up2)
        y_up1_tr_add = layers.Add()([up1_tr, down1])
        y_up1 = layers.Conv2D(filters=num_filters, kernel_size=3, activation="relu", padding="same")(y_up1_tr_add)

        final_conv = layers.Conv2D(filters=6, kernel_size=1)(y_up1)
        x = layers.Activation("softmax")(final_conv)

        model = models.Model(input, x)

        return model