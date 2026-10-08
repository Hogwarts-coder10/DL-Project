import tensorflow as tf
from tensorflow.keras import layers, models


def residual_conv_block(x, filters, n_convs, name):
    """
    Residual convolutional block used in the 2D V-Net-style architecture.

    Each block consists of:
        Conv2D -> BatchNorm -> PReLU
    repeated n_convs times, followed by a residual connection.
    """
    skip = x

    # Match channel dimensions for residual addition if necessary
    if skip.shape[-1] != filters:
        skip = layers.Conv2D(
            filters,
            (1, 1),
            padding="same",
            name=f"{name}_skip"
        )(skip)

    for i in range(n_convs):
        x = layers.Conv2D(
            filters,
            (3, 3),
            padding="same",
            name=f"{name}_conv{i + 1}"
        )(x)

        x = layers.BatchNormalization(
            name=f"{name}_bn{i + 1}"
        )(x)

        x = layers.PReLU(
            shared_axes=[1, 2],
            name=f"{name}_prelu{i + 1}"
        )(x)

    x = layers.Add(name=f"{name}_residual_add")([x, skip])

    return x


def down_block(x, filters, n_convs, name):
    """
    Encoder block:
        Residual convolutional block
        -> learned strided convolution for downsampling
    """
    conv_out = residual_conv_block(
        x,
        filters,
        n_convs,
        name=f"{name}_residual"
    )

    down = layers.Conv2D(
        filters * 2,
        (2, 2),
        strides=(2, 2),
        padding="same",
        name=f"{name}_downsample"
    )(conv_out)

    down = layers.BatchNormalization(
        name=f"{name}_down_bn"
    )(down)

    down = layers.PReLU(
        shared_axes=[1, 2],
        name=f"{name}_down_prelu"
    )(down)

    return conv_out, down


def up_block(x, skip, filters, n_convs, name):
    """
    Decoder block:
        Transposed convolution for upsampling
        -> skip connection concatenation
        -> residual convolutional block
    """
    up = layers.Conv2DTranspose(
        filters,
        (2, 2),
        strides=(2, 2),
        padding="same",
        name=f"{name}_upsample"
    )(x)

    up = layers.BatchNormalization(
        name=f"{name}_up_bn"
    )(up)

    up = layers.PReLU(
        shared_axes=[1, 2],
        name=f"{name}_up_prelu"
    )(up)

    merged = layers.Concatenate(
        name=f"{name}_skip_concat"
    )([up, skip])

    return residual_conv_block(
        merged,
        filters,
        n_convs,
        name=f"{name}_residual"
    )


def build_model(input_shape=(256, 256, 3), num_classes=4):
    """
    Build a 2D V-Net-style segmentation model.

    This is a 2D adaptation of the V-Net architecture for
    RGB lunar terrain images.

    Returns:
        tf.keras.Model
    """

    inputs = layers.Input(
        shape=input_shape,
        name="input_image"
    )

    # ============================================================
    # ENCODER
    # ============================================================

    s1, p1 = down_block(
        inputs,
        filters=64,
        n_convs=1,
        name="encoder_stage1"
    )

    s2, p2 = down_block(
        p1,
        filters=128,
        n_convs=2,
        name="encoder_stage2"
    )

    s3, p3 = down_block(
        p2,
        filters=256,
        n_convs=3,
        name="encoder_stage3"
    )

    s4, p4 = down_block(
        p3,
        filters=512,
        n_convs=3,
        name="encoder_stage4"
    )

    # ============================================================
    # BOTTLENECK
    # ============================================================

    bottleneck = residual_conv_block(
        p4,
        filters=1024,
        n_convs=3,
        name="bottleneck"
    )

    # ============================================================
    # DECODER
    # ============================================================

    d4 = up_block(
        bottleneck,
        s4,
        filters=512,
        n_convs=3,
        name="decoder_stage4"
    )

    d3 = up_block(
        d4,
        s3,
        filters=256,
        n_convs=3,
        name="decoder_stage3"
    )

    d2 = up_block(
        d3,
        s2,
        filters=128,
        n_convs=2,
        name="decoder_stage2"
    )

    d1 = up_block(
        d2,
        s1,
        filters=64,
        n_convs=1,
        name="decoder_stage1"
    )

    # ============================================================
    # OUTPUT
    # ============================================================

    activation = "softmax" if num_classes > 1 else "sigmoid"

    outputs = layers.Conv2D(
        num_classes,
        (1, 1),
        activation=activation,
        name="segmentation_output"
    )(d1)

    return models.Model(
        inputs,
        outputs,
        name="VNet_2D"
    )
