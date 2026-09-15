import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1e-6):
    """
    Calculates the Dice Coefficient.
    The smooth term prevents division by zero if both sets are empty.
    """
    # Cast ground truth to float32 to match predictions
    y_true_f = tf.cast(y_true, tf.float32)
    
    # Flatten tensors using pure TensorFlow (reshaping to 1D)
    y_true_f = tf.reshape(y_true_f, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    
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
