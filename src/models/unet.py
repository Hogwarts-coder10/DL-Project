from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate, BatchNormalization, Activation
from keras.models import Model

def conv_block(x, filters):
    """
    Standard block: Conv2D -> BatchNorm -> ReLU (x2)
    Used repeatedly throughout the encoder and decoder.
    """
    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    
    x = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def build_model(input_shape=(256, 256, 3), num_classes=3):
    """
    Constructs a 4-level U-Net architecture (same depth and filter sizes
    as the UNet++ model, so the two can be compared fairly).
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
    
    # --- Bottleneck ---
    c5 = conv_block(p4, nb_filter[4])
    
    # --- The Decoder (Upsampling) ---
    # Each stage: upsample -> concatenate with the matching encoder feature map
    # (plain skip connection, no nested dense blocks) -> conv block
    u6 = Conv2DTranspose(nb_filter[3], (2,2), strides=(2,2), padding='same')(c5)
    c6 = conv_block(concatenate([u6, c4]), nb_filter[3])
    
    u7 = Conv2DTranspose(nb_filter[2], (2,2), strides=(2,2), padding='same')(c6)
    c7 = conv_block(concatenate([u7, c3]), nb_filter[2])
    
    u8 = Conv2DTranspose(nb_filter[1], (2,2), strides=(2,2), padding='same')(c7)
    c8 = conv_block(concatenate([u8, c2]), nb_filter[1])
    
    u9 = Conv2DTranspose(nb_filter[0], (2,2), strides=(2,2), padding='same')(c8)
    c9 = conv_block(concatenate([u9, c1]), nb_filter[0])
    
    # --- Output Layer ---
    # Using softmax because we have mutually exclusive classes (Sky, Ground, Rocks)
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(c9)
    
    return Model(inputs=[inputs], outputs=[outputs], name="UNet")

if __name__ == "__main__":
    model = build_model()
    model.summary()
