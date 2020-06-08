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
import wandb

#First Party Imports
import utilities
from constants import Constants
from preprocess import Preprocess
from tunableVariables import Tunable


wandb.init(project="hrgan")

class Train():
    """
    Combination of training functions to use in other files
    """

    def __init__(self, preprocessVars):
        """
        Initialization of all variables needed
        """

        self.fullXRes = Tunable.fullIMGXRes
        self.fullYRes = Tunable.fullIMGYRes
        self.chunkXRes = Tunable.chunkIMGXRes
        self.chunkYRes = Tunable.chunkIMGYRes
        self.chunks = Tunable.chunks

        self.EPOCHS = Tunable.totalEPOCHS
        self.latentSize = Tunable.latentSize

        self.trainDataset = preprocessVars.trainDataset
        self.trainingChunk = preprocessVars.trainingChunk

        self.epochGenLoss = [2]
        self.epochDiscLoss = [0]
        self.totalGenLoss = [2]
        self.totalDiscLoss = [0]

        self.discriminatorOptimizer = tf.optimizers.Adam(1e-5)
        self.generatorOptimizer = tf.optimizers.Adam(1e-4)

        self.discriminatorModel = None
        self.generatorModel = None

        self.makeDiscriminatorModel()
        self.makeGeneratorModel()

    def makeDiscriminatorModel(self):
        """
        Makes the discriminator model
        """

        self.discriminatorModel = tf.keras.Sequential()
        self.discriminatorModel.add(tf.keras.layers.Conv2D(Tunable.convFilters, (3, 3), padding="same", input_shape=(self.chunkXRes, self.chunkYRes, Tunable.colorChannels)))
        self.discriminatorModel.add(tf.keras.layers.LeakyReLU())
        self.discriminatorModel.add(tf.keras.layers.Dropout(Tunable.dropout))

        self.discriminatorModel.add(tf.keras.layers.Dense(128, activation="relu"))
        self.discriminatorModel.add(tf.keras.layers.Dropout(Tunable.dropout))

        self.discriminatorModel.add(tf.keras.layers.Flatten())
        self.discriminatorModel.add(tf.keras.layers.Dense(1))

    def makeGeneratorModel(self):
        """
        Making the generator model
        """

        self.generatorModel = tf.keras.Sequential()
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

        self.generatorModel.add(tf.keras.layers.Conv2DTranspose(Tunable.colorChannels, (3, 3), strides=(2, 2), padding="same", use_bias=False, activation="tanh"))

    def train(self):
        """
        Loops through the data and feeds the data to trainStep
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
            for images in self.trainDataset:
                self.trainStep(images, np.loadtxt(fr"C:\Coding\Python\ML\GAN\HR_GAN\latentSpace\noise{epoch}.txt"))
                latentPoint += 1
            end = timer()
            print(f"Chunk Time Left: {str(datetime.timedelta(seconds=int((self.EPOCHS-(epoch+1))*(end-start))))}")
            print(f"Chunk {self.trainingChunk+1}, Epoch {epoch+1}/{self.EPOCHS} at {self.chunkXRes}x{self.chunkYRes} finished in: {(end-start):.4f} sec")
            print(f"The mean generator loss for epoch {epoch+1} is {mean(self.epochGenLoss)}.")
            print(f"The mean discriminator loss for epoch {epoch+1} is {mean(self.epochDiscLoss)}.\n")
            wandb.log({"genLoss": mean(self.epochGenLoss), "discLoss": mean(self.epochDiscLoss), "epoch": epoch+1})
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
            genLoss = utilities.getGeneratorLoss(fakeOutput)
            discLoss = utilities.getDiscriminatorLoss(realOutput, fakeOutput)
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
