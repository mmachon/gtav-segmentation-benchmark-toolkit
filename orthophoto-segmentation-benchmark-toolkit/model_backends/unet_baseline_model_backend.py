from tensorflow.keras import layers, models, optimizers
import numpy as np
import tensorflow as tf

from .model_backend import ModelBackend


class UnetBaselineModelBackend(ModelBackend):

    def __init__(self, chip_size):
        super().__init__(chip_size=chip_size)

    def compile(self):
        model_backend = self.build_unet(size=self.chip_size, encoder="resnet18")
        model_backend.compile(
            optimizer=optimizers.Adam(lr=1e-4),
            loss='categorical_crossentropy',
            metrics=self.metrics
        )
        return model_backend

    def build_unet(self, size=300, basef=64, maxf=512, encoder='resnet50', pretrained=True):
        input = layers.Input((size, size, 3))

        encoder_model = self.make_encoder(input, name=encoder, pretrained=pretrained)

        crosses = []

        for layer in encoder_model.layers:
            # don't end on padding layers
            if type(layer) == layers.ZeroPadding2D:
                continue
            if type(layer.output_shape) == list:
                l_size = layer.output_shape[0][1]
            elif type(layer.output_shape) == tuple:
                l_size = layer.output_shape[1]
            else:
                raise Exception
            idx = self.get_scale_index(size, l_size)
            if idx is None:
                continue
            if idx >= len(crosses):
                crosses.append(layer)
            else:
                crosses[idx] = layer

        x = crosses[-1].output
        for scale in range(len(crosses)-2, -1, -1):
            nf = min(basef * 2**scale, maxf)
            x = self.upscale(x, nf)
            x = self.act(x)
            x = layers.Concatenate()([
                self.pad_to_scale(x, scale, size=size),
                self.pad_to_scale(crosses[scale].output, scale, size=size)
            ])
            x = self.conv(x, nf)
            x = self.act(x)

        x = self.conv(x, 6)
        x = layers.Activation('softmax')(x)

        return models.Model(input, x)

    def make_encoder(self, input, name='resnet50', pretrained=True):
        if name == 'resnet18':
            from classification_models.keras import Classifiers
            ResNet18, _ = Classifiers.get('resnet18')
            model = ResNet18(
                weights='imagenet' if pretrained else None,
                input_tensor=input,
                include_top=False
            )
        elif name == 'resnet50':
            from tensorflow.keras.applications.resnet import ResNet50
            model = ResNet50(
                weights='imagenet' if pretrained else None,
                input_tensor=input,
                include_top=False
            )
        elif name == 'resnet101':
            from tensorflow.keras.applications.resnet import ResNet101
            model = ResNet101(
                weights='imagenet' if pretrained else None,
                input_tensor=input,
                include_top=False
            )
        elif name == 'resnet152':
            from tensorflow.keras.applications.resnet import ResNet152
            model = ResNet152(
                weights='imagenet' if pretrained else None,
                input_tensor=input,
                include_top=False
            )
        elif name == 'vgg16':
            from tensorflow.keras.applications.vgg16 import VGG16
            model = VGG16(
                weights='imagenet' if pretrained else None,
                input_tensor=input,
                include_top=False
            )
        elif name == 'vgg19':
            from tensorflow.keras.applications.vgg19 import VGG19
            model = VGG19(
                weights='imagenet' if pretrained else None,
                input_tensor=input,
                include_top=False
            )
        else:
            raise Exception(f'unknown encoder {name}')

        return model


    def get_scale_index(self, in_size, l_size):
        for i in range(8):
            s_size = in_size // (2 ** i)
            if abs(l_size - s_size) <= 4:
                return i
        return None


    def pad_to_scale(self, x, scale, size=300):
        expected = int(np.ceil(size / (2. ** scale)))
        diff = expected - int(x.shape[1])
        if diff > 0:
            left = diff // 2
            right = diff - left
            x = self.reflectpad(x, (left, right))
        elif diff < 0:
            left = -diff // 2
            right = -diff - left
            x = layers.Cropping2D(((left, right), (left, right)))(x)
        return x


    def reflectpad(self, x, pad):
        return layers.Lambda(lambda x: tf.pad(x, [(0, 0), pad, pad, (0, 0)], 'REFLECT'))(x)


    def upscale(self, x, nf):
        x = layers.UpSampling2D((2, 2))(x)
        x = self.conv(x, nf, kernel_size=(1, 1))
        return x


    def act(self, x):
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        return x


    def conv(self, x, nf, kernel_size=(3, 3), **kwargs):
        padleft = (kernel_size[0] - 1) // 2
        padright = kernel_size[0] - 1 - padleft
        if padleft > 0 or padright > 0:
            x = self.reflectpad(x, (padleft, padright))
        return layers.Conv2D(nf, kernel_size=kernel_size, padding='valid', **kwargs)(x)
