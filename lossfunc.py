import numpy as np
import tensorflow as tf

# def cross_entropy(y_,y,class_weights=None):
#     if class_weights is not None:
#         flat_logits = tf.reshape(y_, [-1, 2])
#         flat_labels = tf.reshape(y, [-1, 1])
#         loss_map = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flat_logits, labels=tf.squeeze(flat_labels,squeeze_dims=[-1]),name="entropy")
#         #loss_map = tf.nn.softmax_cross_entropy_with_logits_v2(logits=flat_logits,
#         #                                                      labels=flat_labels)
#         flat_labels = tf.cast(flat_labels,'float32')
#         class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

#         weight_map = tf.multiply(flat_labels, class_weights)
#         weight_map = tf.reduce_sum(weight_map, axis=1)


#         weighted_loss = tf.multiply(loss_map, weight_map)

#         loss = tf.reduce_mean(weighted_loss)
#         return loss
#     else:
#         return tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_, labels=tf.squeeze(y,squeeze_dims=[-1]),name="entropy")))
        
        
        
def cross_entropy(y_,y,class_weights=None):
    if class_weights is None:
        print(y_)
        labelslol=tf.squeeze(y,squeeze_dims=[-1])
        print(labelslol)
        print(y)
        print('TensorFlow version: {0}'.format(tf.__version__))
        loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_, labels=tf.squeeze(y,squeeze_dims=[-1]),name="entropy")))
        return loss



# def sigmoid_cross_entropy(y_,y,class_weights=None):
#     if class_weights is None:
#         print(y_)
#         print(y)
#         loss = tf.reduce_mean((tf.nn.sigmoid_cross_entropy_with_logits(logits=y_, targets=tf.squeeze(y,squeeze_dims=[-1]),name="entropy")))
#         return loss
        









def dice_loss(y_true, y_pred, class_weights=None):#works
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.nn.softmax(y_pred, axis=-1)

    if class_weights is not None:
        w = tf.constant(np.array(class_weights, dtype=np.float32))
        w = tf.expand_dims(w, axis=0)
        w = tf.expand_dims(w, axis=0)
        y_true = y_true * w

    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denominator = tf.reduce_sum(tf.square(y_true) + tf.square(y_pred), axis=(1,2,3))
    loss = 1 - tf.reduce_mean(numerator / denominator)



    return loss
    
    



def dice_loss_acc_used(y_true, y_pred, class_weights=None):#works
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.nn.softmax(y_pred, axis=-1)

    if class_weights is not None:
        w = tf.constant(np.array(class_weights, dtype=np.float32))
        w = tf.expand_dims(w, axis=0)
        w = tf.expand_dims(w, axis=0)
        y_true = y_true * w

    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1,2,3))
    denominator = tf.reduce_sum(tf.square(y_true) + tf.square(y_pred), axis=(1,2,3))
    dice_loss = 1 - tf.reduce_mean(numerator / denominator)
    
    # Compute the penalty term based on the difference between predicted labels and true labels
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, axis=-1), tf.argmax(y_true, axis=-1)), tf.float32))
    penalty = (1 - accuracy) ** 2
    
    #make penalty 0  for testing sake arham
    # penalty =  0
    
    # Combine the dice loss and penalty term
    loss = dice_loss + penalty
    
    return loss
    
    
    