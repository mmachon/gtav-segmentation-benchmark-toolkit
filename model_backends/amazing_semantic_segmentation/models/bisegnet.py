from tensorflow.keras.layers import *
from tensorflow.keras.applications import *
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.models import Model
from classification_models.keras import Classifiers


def ConvAndBatch(x, n_filters=64, kernel=(2, 2), strides=(1, 1), padding='valid', activation='relu'):
    filters = n_filters

    conv_ = Conv2D(filters=filters,
                   kernel_size=kernel,
                   strides=strides,
                   padding=padding)

    batch_norm = BatchNormalization()

    activation = Activation(activation)

    x = conv_(x)
    x = batch_norm(x)
    x = activation(x)

    return x


def ConvAndAct(x, n_filters, kernel=(1, 1), activation='relu', pooling=False):
    poolingLayer = AveragePooling2D(pool_size=(1, 1), padding='same')
    convLayer = Conv2D(filters=n_filters,
                       kernel_size=kernel,
                       strides=1)

    activation = Activation(activation)

    if pooling:
        x = poolingLayer(x)

    x = convLayer(x)
    x = activation(x)

    return x


def AttentionRefinmentModule(inputs, n_filters):
    filters = n_filters

    poolingLayer = AveragePooling2D(pool_size=(1, 1), padding='same')

    x = poolingLayer(inputs)
    x = ConvAndBatch(x, kernel=(1, 1), n_filters=filters, activation='sigmoid')

    return multiply([inputs, x])


def FeatureFusionModule(input_f, input_s, n_filters):
    concatenate = Concatenate(axis=-1)([input_f, input_s])

    branch0 = ConvAndBatch(concatenate, n_filters=n_filters, kernel=(3, 3), padding='same')
    branch_1 = ConvAndAct(branch0, n_filters=n_filters, pooling=True, activation='relu')
    branch_1 = ConvAndAct(branch_1, n_filters=n_filters, pooling=False, activation='sigmoid')

    x = multiply([branch0, branch_1])
    return Add()([branch0, x])


def ContextPath(layer_13, layer_14, backbone="xception"):
    globalmax = GlobalAveragePooling2D()

    if backbone == "resnet50":
        block1 = AttentionRefinmentModule(layer_13, n_filters=512)
        block2 = AttentionRefinmentModule(layer_14, n_filters=2048)
    elif backbone == "resnet18":
        block1 = AttentionRefinmentModule(layer_13, n_filters=512)
        block2 = AttentionRefinmentModule(layer_14, n_filters=512)
    elif backbone == "xception":
        block1 = AttentionRefinmentModule(layer_13, n_filters=1024)
        block2 = AttentionRefinmentModule(layer_14, n_filters=2048)
    elif backbone == "efficientnetb3":
        block1 = AttentionRefinmentModule(layer_13, n_filters=384)
        block2 = AttentionRefinmentModule(layer_14, n_filters=1536)

    global_channels = globalmax(block2)
    block2_scaled = multiply([global_channels, block2])

    block1 = UpSampling2D(size=(4, 4), interpolation='bilinear')(block1)
    block2_scaled = UpSampling2D(size=(4, 4), interpolation='bilinear')(block2_scaled)

    cnc = Concatenate(axis=-1)([block1, block2_scaled])

    return cnc


def FinalModel(x, layer_13, layer_14, backbone):
    x = ConvAndBatch(x, 32, strides=2)
    x = ConvAndBatch(x, 64, strides=2)
    x = ConvAndBatch(x, 156, strides=2)

    # context path
    cp = ContextPath(layer_13, layer_14, backbone)
    fusion = FeatureFusionModule(cp, x, 32)
    ans = UpSampling2D(size=(8, 8), interpolation='bilinear')(fusion)

    return ans


def get_model(backbone_name="xception"):
    if backbone_name == "xception":
        backbone = Xception(weights='imagenet', input_shape=(384, 384, 3), include_top=False, classes=6)
        tail_prev = backbone.get_layer('block13_pool').output
        inputs = backbone.input
        x = Lambda(lambda image: preprocess_input(image))(inputs)
        tail = backbone.output
        output = FinalModel(x, tail_prev, tail, backbone_name)
    elif backbone_name == "resnet18" or backbone_name == "resnet50":
        if backbone_name == "resnet18":
            ResNet18, _ = Classifiers.get('resnet18')
            backbone = ResNet18(
                weights='imagenet',
                input_tensor=Input((384, 384, 3)),
                include_top=False,
                classes=6
            )
            tail_prev = backbone.get_layer('stage4_unit2_relu2').output
        elif backbone_name == "resnet50":
            backbone = ResNet50(weights='imagenet', input_shape=(384, 384, 3), include_top=False, classes=6)
            tail_prev = backbone.get_layer('conv5_block3_2_relu').output
        inputs = backbone.input
        x = Lambda(lambda image: preprocess_input(image))(inputs)
        tail = backbone.output
        output = FinalModel(x, tail_prev, tail, backbone_name)
    if backbone_name == "efficientnetb3":
       backbone = EfficientNetB3(weights='imagenet', input_shape=(384, 384, 3), include_top=False, classes=6)
       tail_prev = backbone.get_layer('block7b_project_conv').output
       inputs = backbone.input
       x = Lambda(lambda image: preprocess_input(image))(inputs)
       tail = backbone.output
       output = FinalModel(x, tail_prev, tail, backbone_name)


    x = Conv2D(
        filters=6,
        kernel_size=(3, 3),
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        name='final_conv',
    )(output)
    x = Activation("softmax", name="softmax")(x)
    return Model(inputs, x)
