# Keras implementation of SkyScapesNet-Dense
# Paper:
#   S. Azimi, C. Henry, L. Sommer, A. Schaumann, and E. Vig
#   "Skyscapes -- Fine-Grained Semantic Understanding of Aerial Scenes,"
#   International Conference on Computer Vision (ICCV), October 2019.
# https://www.dlr.de/eoc/de/desktopdefault.aspx/tabid-12760/22294_read-58694

from tensorflow.keras import optimizers, losses, models
from tensorflow.keras.layers import Conv2D, ReLU, BatchNormalization, SeparableConv2D, Dropout, MaxPool2D, Input, Concatenate, UpSampling2D, Add, SeparableConvolution2D, AveragePooling2D, Conv2DTranspose, Activation


from .model_backend import ModelBackend


def sl_block(x, filter, dropout):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filter, (3, 3), padding="same")(x)
    x = SeparableConv2D(filter, (3, 3), padding="same")(x)
    x = Dropout(dropout)(x)
    return x


def fdb_block(x, filter, dropout, skip=False):
    sl_1 = sl_block(x, filter, dropout)
    cat_1 = Concatenate()([x, sl_1])
    sl_2 = sl_block(cat_1, filter, dropout)
    cat_2 = Concatenate()([sl_1, cat_1, sl_2])
    sl_3 = sl_block(cat_2, filter, dropout)
    cat_3 = Concatenate()([sl_1, sl_2, sl_3, cat_2])
    if skip:
        return Concatenate()([cat_3, x])
    else:
        return cat_3


def frsr_block(conv_input, maxpool_input, filter):
    x = MaxPool2D()(conv_input)
    x = Concatenate()([x, maxpool_input])
    x = Conv2D(filter, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = SeparableConv2D(filter, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    relu_out = ReLU()(x)
    x = Conv2D(filter, (3, 3), padding="same")(relu_out)
    x = UpSampling2D()(x)

    return x, relu_out


def dos_block(x, filter):
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filter, (1, 1), padding="same")(x)
    x = SeparableConv2D(filter, (3, 3), padding="same")(x)
    x = MaxPool2D()(x)
    return x


def lkbr_block(input, filter):
    x = Conv2D(filter, (3, 1), padding="same")(input)
    x = Conv2D(filter, (1, 3), padding="same")(x)

    y = Conv2D(filter, (1, 3), padding="same")(input)
    y = Conv2D(filter, (3, 1), padding="same")(y)
    add_1 = Add()([input, x, y])
    x = Conv2D(filter, (3, 3), padding="same")(add_1)
    x = SeparableConv2D(filter, (3, 3), padding="same")(x)
    x = ReLU()(x)
    add_2 = Add()([add_1, x])
    cat = Concatenate()([add_2, input])
    return cat


def craspp_block(x, filter):
    image_pooling = AveragePooling2D(1, 1)(x)
    atrous_18 = Conv2D(filter, (3, 3), dilation_rate=(18, 18), padding="same")(x)
    atrous_12 = Conv2D(filter, (3, 3), dilation_rate=(12, 12), padding="same")(x)
    atrous_6 = Conv2D(filter, (3, 3), dilation_rate=(6, 6), padding="same")(x)
    conv_1 = Conv2D(filter, (1, 1), padding="same")(x)

    cat_1 = Concatenate()([image_pooling, atrous_18])
    cat_2 = Concatenate()([atrous_12, cat_1])
    cat_3 = Concatenate()([atrous_6, cat_2])
    cat_4 = Concatenate()([conv_1, cat_3])

    conv_2 = Conv2D(filter, (1, 1), padding="same")(cat_1)
    atrous_6_2 = Conv2D(filter, (3, 3), dilation_rate=(6, 6), padding="same")(cat_2)
    atrous_12_2 = Conv2D(filter, (3, 3), dilation_rate=(12, 12), padding="same")(cat_3)
    atrous_18_2 = Conv2D(filter, (3, 3), dilation_rate=(18, 18), padding="same")(cat_4)

    cat_5 = Concatenate()([conv_2, atrous_18, atrous_12, atrous_6_2])
    cat_6 = Concatenate()([atrous_6, conv_1, atrous_12_2, atrous_18_2])
    add = Add()([cat_5, cat_6])
    return add


def ups_block(x, filter, dropout):
    x = Conv2DTranspose(filter, (3, 3), strides=(2, 2), padding="same")(x)
    x = Dropout(dropout)(x)
    return x


class SkyScapesNetBackend(ModelBackend):

    def __init__(self, chip_size):
        super().__init__(chip_size)

    def compile(self):
        model_backend = self.build_skyscapesnet()
        model_backend.compile(
            optimizer=optimizers.Adam(lr=1e-4),
            loss=losses.CategoricalCrossentropy(),
            metrics=self.metrics
        )
        model_backend.summary()
        return model_backend

    def build_skyscapesnet(self, filter=48, dropout=0.5, classes=6):
        input = Input((self.chip_size, self.chip_size, 3))

        maxpool = MaxPool2D()(input)
        conv = Conv2D(filter, (3, 3), padding="same")(input)
        frsr_1, frsr_1_relu = frsr_block(conv, maxpool, filter)
        fdb_1 = fdb_block(conv, filter, dropout)
        cat_1 = Concatenate()([frsr_1, fdb_1])
        lkbr_1 = lkbr_block(cat_1, filter*8)
        dos_1 = dos_block(cat_1, filter)
        cat_2 = Concatenate()([dos_1, frsr_1_relu])
        maxpool = MaxPool2D()(cat_2)
        conv = Conv2D(filter, (3, 3), padding="same")(cat_2)
        fdb_2 = fdb_block(cat_2, filter, dropout)
        frsr_2, frsr_2_relu = frsr_block(conv, maxpool, filter)
        cat_3 = Concatenate()([frsr_2, fdb_2])
        lkbr_2 = lkbr_block(cat_3, filter*9)
        dos_2 = dos_block(cat_3, filter)
        cat_4 = Concatenate()([frsr_2_relu, dos_2])
        fdb_skip = fdb_block(cat_4, filter, dropout, skip=True)
        craspp = craspp_block(fdb_skip, filter)
        cat_5 = Concatenate()([cat_4, craspp])
        ups_1 = ups_block(cat_5, filter, dropout)
        cat_6 = Concatenate()([ups_1, lkbr_2])
        fdb_3 = fdb_block(cat_6, filter, dropout)
        ups_2 = ups_block(fdb_3, filter, dropout)
        cat_7 = Concatenate()([ups_2, lkbr_1])
        fdb_4 = fdb_block(cat_7, filter, dropout)

        final_conv = Conv2D(filters=classes, kernel_size=1)(fdb_4)
        x = Activation("softmax")(final_conv)
        model = models.Model(input, x)
        return model

