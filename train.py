#pylint: disable=C0103, C0301
"""
Trains the a model given the dimesntions of traingin images,
epochs and latent size
"""
__author__ = "Noupin"

#Third Party Imports
import datetime
from statistics import mean
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np


class Train():
    """
    Combination of training functions to use in other files
    """

    def __init__(self, commonVars, preproVars):
        """
        Initialization of all variables needed
        """

        #Resolutions & Amount of Chunks
        self.fullXRes = commonVars.fullIMGXRes
        self.fullYRes = commonVars.fullIMGYRes
        self.chunkXRes = commonVars.chunkIMGXRes
        self.chunkYRes = commonVars.chunkIMGYRes
        self.chunks = int((self.fullXRes**2)/(self.chunkXRes**2))

        #Setting the number of EPOCHS and size of latent space
        self.EPOCHS = commonVars.totalEPOCHS
        self.latentSize = commonVars.latentSize

        #Setting the training dataset and the current chunk
        self.trainDataset = preproVars.trainDataset
        self.trainingChunk = preproVars.trainingChunk

        #Setting up list for average loss of epochs and chunks
        self.epochGenLoss = [2]
        self.epochDiscLoss = [0]
        self.totalGenLoss = [2]
        self.totalDiscLoss = [0]

        #Setting the generator and discriminators optimizers
        self.discriminatorOptimizer = tf.optimizers.Adam()
        self.generatorOptimizer = tf.optimizers.Adam()
        #self.discriminatorOptimizer = tf.optimizers.RMSprop(1e-4)
        #self.generatorOptimizer = tf.optimizers.RMSprop(1e-5)

        #initializing discriminator and gnerator variables
        self.discriminatorModel = None
        self.generatorModel = None

        #Making Discriminator & Generator
        self.makeDiscriminatorModel()
        self.makeGeneratorModel()

    def makeDiscriminatorModel(self):
        """
        Makes the discriminator model
        """

        self.discriminatorModel = tf.keras.Sequential()
        self.discriminatorModel.add(tf.keras.layers.Conv2D(7, (3, 3), padding="same", input_shape=(self.chunkXRes, self.chunkYRes, 3)))
        self.discriminatorModel.add(tf.keras.layers.Flatten())
        self.discriminatorModel.add(tf.keras.layers.LeakyReLU())
        self.discriminatorModel.add(tf.keras.layers.Dropout(0.4))

        self.discriminatorModel.add(tf.keras.layers.Dense(128, activation="relu"))
        self.discriminatorModel.add(tf.keras.layers.Dropout(0.4))
        self.discriminatorModel.add(tf.keras.layers.Dense(128, activation="relu"))
        self.discriminatorModel.add(tf.keras.layers.Dropout(0.4))
        self.discriminatorModel.add(tf.keras.layers.Dense(64, activation="relu"))
        self.discriminatorModel.add(tf.keras.layers.Dropout(0.4))

        self.discriminatorModel.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    def makeGeneratorModel(self):
        """
        Making the generator model
        """

        self.generatorModel = tf.keras.Sequential()
        #7*7*256 for 28x28 img and 14*14*256 for 56x56 image so 1 for a 4x4 img
        self.generatorModel.add(tf.keras.layers.Dense(int(self.chunkXRes/4)*int(self.chunkYRes/4)*256, use_bias=False, input_shape=(self.latentSize,)))
        self.generatorModel.add(tf.keras.layers.BatchNormalization())
        self.generatorModel.add(tf.keras.layers.LeakyReLU())

        self.generatorModel.add(tf.keras.layers.Reshape((int(self.chunkXRes/4), int(self.chunkYRes/4), 256)))

        self.generatorModel.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), padding="same", use_bias=False))
        self.generatorModel.add(tf.keras.layers.BatchNormalization())
        self.generatorModel.add(tf.keras.layers.LeakyReLU())

        self.generatorModel.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", use_bias=False))
        self.generatorModel.add(tf.keras.layers.BatchNormalization())
        self.generatorModel.add(tf.keras.layers.LeakyReLU())

        self.generatorModel.add(tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding="same", use_bias=False, activation="tanh"))

    def train(self):
        """
        Loops throught eh data and feeds the data to trainStep
        BATCH_SIZE at a time
        """

        print("Training Started.")
        chunkStart = timer()
        latentPoint = 0
        for epoch in range(self.EPOCHS):
            self.epochDiscLoss.clear()
            self.epochGenLoss.clear()
            self.epochDiscLoss.append(0)
            self.epochGenLoss.append(2)
            start = timer()
            for images in self.trainDataset:#image.shape is (some #, xRes, yRes, colorChannels) the some # is the BATCH_SIZE
                #Total trainSteps = (Dataset size or total loaded images/BATCH_SIZE)*epochs
                self.trainStep(images, np.loadtxt(fr"C:\Coding\Python\ML\GAN\HR_GAN\latentSpace\noise{epoch}.txt"))
                latentPoint += 1
            end = timer()
            print(f"Chunk Time Left: {str(datetime.timedelta(seconds=int((self.EPOCHS-(epoch+1))*(end-start))))}")
            print(f"Chunk {self.trainingChunk+1}, Epoch {epoch+1}/{self.EPOCHS} at {self.chunkXRes}x{self.chunkYRes} finished in: {(end-start):.4f} sec")
            print(f"The mean generator loss for epoch {epoch+1} is {mean(self.epochGenLoss)}.")
            print(f"The mean discriminator loss for epoch {epoch+1} is {mean(self.epochDiscLoss)}.\n")
        chunkEnd = timer()
        print(f"Chunk {self.trainingChunk+1} finished in: {str(datetime.timedelta(seconds=int(chunkEnd-chunkStart)))}")
        print(f"The total mean generator loss for chunk {self.trainingChunk+1} is {mean(self.totalGenLoss)}.")
        print(f"The total mean discriminator loss for chunk {self.trainingChunk+1} is {mean(self.totalDiscLoss)}.\n\n")

    def trainStep(self, images, fakeImageNoise):
        """
        Getting gradients and applying the corrections to the network
        """

        with tf.GradientTape() as genTape, tf.GradientTape() as discTape:
            generatedImages = self.generatorModel(fakeImageNoise)
            realOutput = self.discriminatorModel(images)
            fakeOutput = self.discriminatorModel(generatedImages)
            #Getting gradient or loss
            genLoss = getGeneratorLoss(fakeOutput)
            discLoss = getDiscriminatorLoss(realOutput, fakeOutput)
            #Find gradient
            gradientsOfGen = genTape.gradient(genLoss, self.generatorModel.trainable_variables)
            gradientsOfDisc = discTape.gradient(discLoss, self.discriminatorModel.trainable_variables)
            #Using optimizer
            self.generatorOptimizer.apply_gradients(zip(gradientsOfGen, self.generatorModel.trainable_variables))
            self.discriminatorOptimizer.apply_gradients(zip(gradientsOfDisc, self.discriminatorModel.trainable_variables))

            self.epochDiscLoss.append(np.mean(discLoss))
            self.epochGenLoss.append(np.mean(genLoss))
            self.totalDiscLoss.append(np.mean(discLoss))
            self.totalGenLoss.append(np.mean(genLoss))

    def main(self):
        """
        Saving the generator and discriminator models aswell as
        checking if the training for all chunks is finished to generate
        and show the final images.
        """

        fullStart = timer()
        self.train()
        self.discriminatorModel.save(fr"C:\Coding\Models\ganModels\HighRes\DISC\chunk{self.trainingChunk+1}of{self.chunks}"+
                                     fr"landscapeDISC{self.fullXRes}x{self.fullYRes}res-{self.EPOCHS}epochs-{self.latentSize}latent.model")
        self.generatorModel.save(fr"C:\Coding\Models\ganModels\HighRes\GEN\chunk{self.trainingChunk+1}of{self.chunks}"+
                                 fr"landscapeGEN{self.fullXRes}x{self.fullYRes}res-{self.EPOCHS}epochs-{self.latentSize}latent.model")
        fullStop = timer()
        print(f"Training {self.EPOCHS} epochs finished in: {str(datetime.timedelta(seconds=int(fullStop-fullStart)))}")


#No-self-use functions
def getDiscriminatorLoss(realPredictions, fakePredictions):
    """
    Getting the discriminator loss through a
    binary-crossentropy loss function
    """
    #Could need try: except statement
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
