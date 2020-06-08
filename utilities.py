"""
No-self-use functions
"""

#Third Party Imports
import tensorflow as tf

def factors(num, cap=None):
    """
    Get factors of a number below the cap if given
    """
    retLst = []
    for factor in range(1, num+1):
        if num % factor == 0:
            retLst.append(factor)
        if factor == cap:
            break
    return retLst

def getDiscriminatorLoss(realPredictions, fakePredictions):
    """
    Getting the discriminator loss through a
    binary-crossentropy loss function
    """
    
    realPredictions = tf.math.sigmoid(realPredictions)
    fakePredictions = tf.math.sigmoid(fakePredictions)
    realLoss = tf.losses.binary_crossentropy(tf.ones_like(realPredictions), realPredictions)
    fakeLoss = tf.losses.binary_crossentropy(tf.zeros_like(fakePredictions), fakePredictions)
    return fakeLoss+realLoss

def getGeneratorLoss(fakePredictions):
    """
    Getting the generator loss through a
    binary-crossentropy loss function
    """
    fakePredictions = tf.math.sigmoid(fakePredictions)
    fakeLoss = tf.losses.binary_crossentropy(tf.ones_like(fakePredictions), fakePredictions)
    return fakeLoss

def getWDiscriminatorLoss(realPredictions, fakePredictions):
    """
    Getting the discriminator loss through a
    wasserstien loss function
    """
    return tf.reduce_mean(realPredictions) - tf.reduce_mean(fakePredictions)

def getWGeneratorLoss(fakePredictions):
    """
    Getting the generator loss through a
    wasserstien loss function
    """
    return -tf.reduce_mean(fakePredictions)
