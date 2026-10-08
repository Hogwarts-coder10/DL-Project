import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Layer
from keras.models import Model


class MaxPoolingWithArgmax2D(Layer):
    """
    2x2 max pooling that ALSO returns the argmax indices.
    SegNet's signature trick: the decoder reuses these indices to unpool.
    """
    def call(self, x):
        pooled, indices = tf.nn.max_pool_with_argmax(
            x, ksize=2, strides=2, padding="SAME", include_batch_in_index=True
        )
        return pooled, indices

    def compute_output_shape(self, input_shape):
        b, h, w, c = input_shape
        out = (b, None if h is None else (h + 1) // 2, None if w is None else (w + 1) // 2, c)
        return out, out


class MaxUnpooling2D(Layer):
    """
    Places each pooled value back at the position it came from (stored indices);
    every other position is zero. Output is 2x the spatial size of the input.
    """
    def call(self, inputs):
        pooled, indices = inputs
        shape = tf.shape(pooled)
        b, h, w, c = shape[0], shape[1], shape[2], shape[3]
        flat_size = tf.cast(b * h * 2 * w * 2 * c, indices.dtype)

        out = tf.scatter_nd(
            tf.expand_dims(tf.reshape(indices, [-1]), 1),
            tf.reshape(pooled, [-1]),
            tf.reshape(flat_size, [1]),
        )
        return tf.reshape(out, [b, h * 2, w * 2, c])

    def compute_output_shape(self, input_shape):
        b, h, w, c = input_shape[0]
        return (b, None if h is None else h * 2, None if w is None else w * 2, c)


def conv_block(x, filters, num_convs=2):
    """
    Standard block: (Conv2D -> BatchNorm -> ReLU) x num_convs
    Used repeatedly throughout the encoder and decoder.
    """
    for _ in range(num_convs):
        x = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x


def build_model(input_shape=(256, 256, 3), num_classes=3):
    """
    Constructs a SegNet architecture (VGG16-style encoder, 5 levels).
    Input height/width must be divisible by 32.
    """
    inputs = Input(input_shape)

    # Filter sizes for each depth level
    nb_filter = [64, 128, 256, 512, 512]

    # --- The Encoder (Downsampling, indices saved) ---
    c1 = conv_block(inputs, nb_filter[0], 2)
    p1, idx1 = MaxPoolingWithArgmax2D()(c1)

    c2 = conv_block(p1, nb_filter[1], 2)
    p2, idx2 = MaxPoolingWithArgmax2D()(c2)

    c3 = conv_block(p2, nb_filter[2], 3)
    p3, idx3 = MaxPoolingWithArgmax2D()(c3)

    c4 = conv_block(p3, nb_filter[3], 3)
    p4, idx4 = MaxPoolingWithArgmax2D()(c4)

    c5 = conv_block(p4, nb_filter[4], 3)
    p5, idx5 = MaxPoolingWithArgmax2D()(c5)

    # --- The Decoder (Unpooling with saved indices, no skip concatenation) ---
    u5 = MaxUnpooling2D()([p5, idx5])
    d5 = conv_block(u5, nb_filter[4], 3)

    u4 = MaxUnpooling2D()([d5, idx4])
    d4 = conv_block(u4, nb_filter[3], 3)

    u3 = MaxUnpooling2D()([d4, idx3])
    d3 = conv_block(u3, nb_filter[2], 3)

    u2 = MaxUnpooling2D()([d3, idx2])
    d2 = conv_block(u2, nb_filter[1], 2)

    u1 = MaxUnpooling2D()([d2, idx1])
    d1 = conv_block(u1, nb_filter[0], 2)

    # --- Output Layer ---
    # Using softmax because we have mutually exclusive classes (Sky, Ground, Rocks)
    outputs = Conv2D(num_classes, (1, 1), activation="softmax")(d1)

    return Model(inputs=[inputs], outputs=[outputs], name="SegNet")
