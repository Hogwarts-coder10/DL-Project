import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Calculates the Dice Coefficient.
    The smooth term prevents division by zero if both sets are empty.
    """
    # Cast both inputs to float32 for mixed precision compatibility
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred, tf.float32)
    
    # Flatten tensors using pure TensorFlow (reshaping to 1D)
    y_true_f = tf.reshape(y_true_f, [-1])
    y_pred_f = tf.reshape(y_pred_f, [-1])
    
    # Calculate the intersection (X ∩ Y)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    
    # Calculate the Dice score
    sum_true = tf.reduce_sum(y_true_f)
    sum_pred = tf.reduce_sum(y_pred_f)
    
    return (2. * intersection + smooth) / (sum_true + sum_pred + smooth)

def dice_loss(y_true, y_pred):
    """
    Dice Loss is simply 1 minus the Dice Coefficient.
    Minimizing this loss maximizes the overlap.
    """
    return 1.0 - dice_coef(y_true, y_pred)

def cce_loss(y_true,y_pred):
    """
    Combined Categorial Cross Entropy + Dice loss
    CCE provides smooth non-zero gradients for minority classes to prevent collapse.
    """

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    cce_loss = tf.reduce_mean(cce)
    d_loss = dice_loss(y_true, y_pred)

    return cce_loss + d_loss
