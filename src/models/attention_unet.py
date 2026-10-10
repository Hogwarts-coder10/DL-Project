import tensorflow as tf
from tensorflow.keras import layers, models

def conv_block(x, filters):
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def attention_gate(g, s, inter_channels):
    theta_x = layers.Conv2D(inter_channels, (2, 2), strides=(2, 2), padding="same")(s)
    phi_g = layers.Conv2D(inter_channels, (1, 1), padding="same")(g)

    f = layers.Activation("relu")(layers.add([theta_x, phi_g]))
    psi_f = layers.Conv2D(1, (1, 1), padding="same")(f)
    psi_f = layers.Activation("sigmoid")(psi_f)

    upsampled_psi = layers.UpSampling2D(size=(2, 2))(psi_f)
    return layers.multiply([s, upsampled_psi])

def build_model(input_shape=(256, 256, 3), num_classes=4, base_filters=32):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = conv_block(inputs, base_filters)             # 32
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, base_filters * 2)         # 64
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, base_filters * 4)         # 128
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, base_filters * 8)         # 256
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    b = conv_block(p4, base_filters * 16)         # 512

    # Decoder
    u1 = layers.Conv2DTranspose(base_filters * 8, (2, 2), strides=(2, 2), padding="same")(b)
    a1 = attention_gate(g=b, s=c4, inter_channels=base_filters * 4)
    d1 = layers.concatenate([u1, a1])
    c5 = conv_block(d1, base_filters * 8)

    u2 = layers.Conv2DTranspose(base_filters * 4, (2, 2), strides=(2, 2), padding="same")(c5)
    a2 = attention_gate(g=c5, s=c3, inter_channels=base_filters * 2)
    d2 = layers.concatenate([u2, a2])
    c6 = conv_block(d2, base_filters * 4)

    u3 = layers.Conv2DTranspose(base_filters * 2, (2, 2), strides=(2, 2), padding="same")(c6)
    a3 = attention_gate(g=c6, s=c2, inter_channels=base_filters)
    d3 = layers.concatenate([u3, a3])
    c7 = conv_block(d3, base_filters * 2)

    u4 = layers.Conv2DTranspose(base_filters, (2, 2), strides=(2, 2), padding="same")(c7)
    a4 = attention_gate(g=c7, s=c1, inter_channels=base_filters // 2)
    d4 = layers.concatenate([u4, a4])
    c8 = conv_block(d4, base_filters)

    # Output Layer forced to FP32
    activation = "softmax" if num_classes > 1 else "sigmoid"
    outputs = layers.Conv2D(num_classes, (1, 1), activation=activation, dtype='float32', name='final_output')(c8)

    return models.Model(inputs, outputs, name="Attention_UNet")
