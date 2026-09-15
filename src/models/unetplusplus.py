from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation
from keras.models import Model

def conv_block(x, filters):
    """
    Standard block: Conv2D -> BatchNorm -> ReLU
    Used repeatedly throughout the encoder and decoder.
    """
    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def build_unet_plus_plus(input_shape=(256, 256, 3), num_classes=3):
    """
    Constructs a 4-level UNet++ architecture.
    """
    inputs = Input(input_shape)
    
    # Filter sizes for each depth level
    nb_filter = [32, 64, 128, 256, 512]
    
    # --- The Encoder (Downsampling) ---
    c1 = conv_block(inputs, nb_filter[0])
    p1 = MaxPooling2D((2,2))(c1)
    
    c2 = conv_block(p1, nb_filter[1])
    p2 = MaxPooling2D((2,2))(c2)
    
    c3 = conv_block(p2, nb_filter[2])
    p3 = MaxPooling2D((2,2))(c3)
    
    c4 = conv_block(p3, nb_filter[3])
    p4 = MaxPooling2D((2,2))(c4)
    
    c5 = conv_block(p4, nb_filter[4])
    
    # --- Nested Skip Pathways (The "++" Dense Blocks) ---
    # Level 1 Dense
    u1_1 = Conv2DTranspose(nb_filter[0], (2,2), strides=(2,2), padding='same')(c2)
    c1_1 = conv_block(concatenate([u1_1, c1]), nb_filter[0])
    
    # Level 2 Dense
    u2_1 = Conv2DTranspose(nb_filter[1], (2,2), strides=(2,2), padding='same')(c3)
    c2_1 = conv_block(concatenate([u2_1, c2]), nb_filter[1])
    
    u1_2 = Conv2DTranspose(nb_filter[0], (2,2), strides=(2,2), padding='same')(c2_1)
    c1_2 = conv_block(concatenate([u1_2, c1, c1_1]), nb_filter[0])
    
    # Level 3 Dense
    u3_1 = Conv2DTranspose(nb_filter[2], (2,2), strides=(2,2), padding='same')(c4)
    c3_1 = conv_block(concatenate([u3_1, c3]), nb_filter[2])
    
    u2_2 = Conv2DTranspose(nb_filter[1], (2,2), strides=(2,2), padding='same')(c3_1)
    c2_2 = conv_block(concatenate([u2_2, c2, c2_1]), nb_filter[1])
    
    u1_3 = Conv2DTranspose(nb_filter[0], (2,2), strides=(2,2), padding='same')(c2_2)
    c1_3 = conv_block(concatenate([u1_3, c1, c1_1, c1_2]), nb_filter[0])
    
    # Level 4 Dense
    u4_1 = Conv2DTranspose(nb_filter[3], (2,2), strides=(2,2), padding='same')(c5)
    c4_1 = conv_block(concatenate([u4_1, c4]), nb_filter[3])
    
    u3_2 = Conv2DTranspose(nb_filter[2], (2,2), strides=(2,2), padding='same')(c4_1)
    c3_2 = conv_block(concatenate([u3_2, c3, c3_1]), nb_filter[2])
    
    u2_3 = Conv2DTranspose(nb_filter[1], (2,2), strides=(2,2), padding='same')(c3_2)
    c2_3 = conv_block(concatenate([u2_3, c2, c2_1, c2_2]), nb_filter[1])
    
    u1_4 = Conv2DTranspose(nb_filter[0], (2,2), strides=(2,2), padding='same')(c2_3)
    c1_4 = conv_block(concatenate([u1_4, c1, c1_1, c1_2, c1_3]), nb_filter[0])
    
    # --- Output Layer ---
    # Using softmax because we have mutually exclusive classes (Sky, Ground, Rocks)
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c1_4)
    
    return Model(inputs=[inputs], outputs=[outputs], name="UNet_Plus_Plus")
