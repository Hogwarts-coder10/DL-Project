import tensorflow as tf
from tensorflow.keras import layers, models

def conv_block(x, filters):
    """Standard double convolution block with Batch Normalization."""
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def attention_gate(g, s, inter_channels):
    """
    Additive Attention Gate.
    g: Gating signal from the deeper, coarser layer (e.g., 16x16)
    s: Skip connection from the encoder at the finer scale (e.g., 32x32)
    """
    # Align channel dimensions and downsample spatial resolution of skip connection
    theta_x = layers.Conv2D(inter_channels, (2, 2), strides=(2, 2), padding="same")(s)
    phi_g = layers.Conv2D(inter_channels, (1, 1), padding="same")(g)

    # Combine signals and apply non-linearity
    f = layers.Activation("relu")(layers.add([theta_x, phi_g]))
    psi_f = layers.Conv2D(1, (1, 1), padding="same")(f)
    psi_f = layers.Activation("sigmoid")(psi_f)

    # Upsample attention coefficient map to match skip connection spatial resolution
    upsampled_psi = layers.UpSampling2D(size=(2, 2))(psi_f)

    # Scale the skip connection features
    return layers.multiply([s, upsampled_psi])

def build_model(input_shape=(256, 256, 3), num_classes=4):
    """Builds the complete Attention U-Net architecture."""
    inputs = layers.Input(shape=input_shape)

    # --- Encoder ---
    c1 = conv_block(inputs, 64)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = conv_block(p3, 512)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # --- Bottleneck ---
    b = conv_block(p4, 1024)

    # --- Decoder with Corrected Attention Gates ---
    # Block 1: Gating signal is b (16x16), skip connection is c4 (32x32)
    u1 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(b)
    a1 = attention_gate(g=b, s=c4, inter_channels=256)
    d1 = layers.concatenate([u1, a1])
    c5 = conv_block(d1, 512)

    # Block 2: Gating signal is c5 (32x32), skip connection is c3 (64x64)
    u2 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c5)
    a2 = attention_gate(g=c5, s=c3, inter_channels=128)
    d2 = layers.concatenate([u2, a2])
    c6 = conv_block(d2, 256)

    # Block 3: Gating signal is c6 (64x64), skip connection is c2 (128x128)
    u3 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c6)
    a3 = attention_gate(g=c6, s=c2, inter_channels=64)
    d3 = layers.concatenate([u3, a3])
    c7 = conv_block(d3, 128)

    # Block 4: Gating signal is c7 (128x128), skip connection is c1 (256x256)
    u4 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c7)
    a4 = attention_gate(g=c7, s=c1, inter_channels=32)
    d4 = layers.concatenate([u4, a4])
    c8 = conv_block(d4, 64)

    # --- Output Layer ---
    activation = "softmax" if num_classes > 1 else "sigmoid"
    outputs = layers.Conv2D(num_classes, (1, 1), activation=activation)(c8)

    return models.Model(inputs, outputs, name="Attention_UNet")
