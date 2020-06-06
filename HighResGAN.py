#pylint: disable=C0103, E1101, C0301, W0105
"""
Makes, trains and shows the images from a GAN. The generated
images can be chunked to allowe for better perfromance and
higher resolution generated images than with a single model.
"""

#Third Party Imports
import os
import random
import math
import datetime
import json
from statistics import mean
from timeit import default_timer as timer
import cv2
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#First Party Imports
#import common


class GAN():
    """
    Given a full image resolution, chunked image resolution and
    a number of epochs, GAN will make the models needed, train them
    and once the training for all chunks are finished output as
    many final images as needed.
    """

    def __init__(self, commonVars):
        """
        Initialization of all variables needed
        """
        super().__init__()
        #480p 2.39:1 720x302
        #2048x2048 is more than 7.3GB of vRAM for the Master DISC model

        #Resolutions & Amount of Chunks
        self.fullXRes = commonVars.fullIMGXRes
        self.fullYRes = commonVars.fullIMGYRes
        self.chunkXRes = commonVars.chunkIMGXRes
        self.chunkYRes = commonVars.chunkIMGYRes
        self.chunks = int((self.fullXRes**2)/(self.chunkXRes**2))

        #Getting the current training chunk and setting up the next trainging chunk
        self.trainingChunk = None
        with open(r"C:\Coding\Python\ML\GAN\trainingChunk.txt") as trainingChunkFile:
            self.trainingChunk = json.load(trainingChunkFile) + 1
        with open(r"C:\Coding\Python\ML\GAN\trainingChunk.txt", "w") as trainingChunkFile:
            json.dump(self.trainingChunk, trainingChunkFile)

        #Dataset Paths
        astro = r"Coding\Python\ML\astroPics"
        cat = r"Datasets\PetImages\Cat"
        dog = r"Datasets\PetImages\Dog"
        kk = r"Datasets\PetImages\KeeKeeKats"
        faces = r"Coding\Python\ML\faces\images"
        S15 = r"RvB Videos\S15\E1"
        landscapes = r"Datasets\Landscapes"

        #List of Datasets
        self.lstOfDatasets = [S15, astro, cat, dog, faces, kk, landscapes]

        #Setting up list for avergae loss of epochs and chunks
        self.epochGenLoss = [2]
        self.epochDiscLoss = [0]
        self.totalGenLoss = [2]
        self.totalDiscLoss = [0]

        #Setting the number of EPOCHS and size of latent space
        self.EPOCHS = commonVars.totalEPOCHS
        self.latentSize = 100

        #Setting up chosen dataset to be used
        self.trainingData = []
        if self.trainingChunk == 0: #Only setup and chunk images on the first chunk
            trainImages = self.setupData(self.lstOfDatasets[6], sizeOf=1)
            print(f"\nStoring {str(int(trainImages[0].shape[0]))} Pictures.")

            #Dump the dataset size so it can be known before the dataset is loaded
            with open(r"C:\Coding\Python\ML\GAN\datasetSize.txt", "w") as datasetSizeFile:
                json.dump(int(trainImages[0].shape[0]), datasetSizeFile)

            #Normalizing the data to be used
            for chunk in range(self.chunks):
                #Bring colors from 255 base to 0-1 floating point base
                trainImages[chunk] = trainImages[chunk]/255.0
                #Reshape to (BATCH_SIZE, chunkX, chunkY, 3)
                trainImages[chunk] = trainImages[chunk].reshape(trainImages[0].shape[0], self.chunkXRes, self.chunkYRes, 3)

            for chunk in range(self.chunks):
                for img in range(int(trainImages[0].shape[0])):
                    #Dump each chunk 0 of the dataset then each chunk 1...
                    np.save(fr"D:\ML\chunkedImgArrays\chunkedImg{(chunk*trainImages[0].shape[0])+img}.npy", trainImages[chunk][img])
            del trainImages
            print("Pictures Stored.")

        #Get the size of the given dataset
        with open(r"C:\Coding\Python\ML\GAN\datasetSize.txt") as imgAmountFile:
            self.datasetSize = json.load(imgAmountFile)

        #Set the BATCH_SIZE by picking a factor format the dataset size
        self.BATCH_SIZE = factors(int(self.datasetSize/4), cap=10)[-1]*4 #Multiplied by 4 since there are 4 version of each image becuase of the composited images
        self.trainDataset = [] #TensorFlow readable training data
        self.trainImages = [] #Images that have been preprocessed and chunked

        print(f"\nLoading {((self.trainingChunk+1)*self.datasetSize)-(self.trainingChunk*self.datasetSize)} Pictures.")
        #Append the images from chunk=trainingChunk to the trainImages
        for img in range(self.trainingChunk*self.datasetSize, (self.trainingChunk+1)*self.datasetSize):
            self.trainImages.append(np.load(fr"D:\ML\chunkedImgArrays\chunkedImg{img}.npy"))

        #Convert the trainImages nested array into a nested array that tensorflow can understand
        self.trainDataset = tf.data.Dataset.from_tensor_slices(self.trainImages).batch(int(self.BATCH_SIZE))
        print("Pictures Loaded.")

        #Giving the user data about the dataset
        print(f"\n\nThe Dataset size is: {int(self.datasetSize/4)} images and {self.datasetSize} color composited images\n\nThe Chunk size is {self.chunkXRes}x"+
              f"{self.chunkYRes} with {self.chunks} chunks\n\nThere are {self.datasetSize*self.chunks} total chunked images\n\nThe BATCH_SIZE is: {self.BATCH_SIZE}"+
              f"\n\nTotal Resolution: {self.chunkXRes}x{self.chunkYRes}\n\n")

        #Creating and writing the latent arrays to file son the first chunk
        if self.trainingChunk == 0:#Only create latent space on first chunks to keep training consistent
            print(f"\nCreating {int((self.datasetSize/self.BATCH_SIZE)*self.EPOCHS)} Latent Training Points.")
            for i in range(int((self.datasetSize/self.BATCH_SIZE)*self.EPOCHS)):#Total trainSteps = (Dataset size or total loaded images/BATCH_SIZE)*epochs
                np.savetxt(fr"C:\Coding\Python\ML\GAN\latentSpace\noise{i}.txt", np.random.randn(self.BATCH_SIZE, self.latentSize))#Save random latent point in file
            print("Finished Creating Latenet Training Points.")

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

    def imgSlice(self, image):
        """
        Slicing the images into a certain size
        """
        slicedImg = []
        width, height = image.size
        upper = 0
        slicesY = int(math.ceil(height/self.chunkXRes))
        slicesX = int(math.ceil(width/self.chunkYRes))

        countY = 1
        for _ in range(slicesY):
            #if we are at the end, set the lower bound to be the bottom of the image
            if countY == slicesY:
                lower = height
            else:
                lower = int(countY * self.chunkYRes)
            countX = 1
            left = 0
            for _ in range(slicesX):
                if countX == slicesX:
                    right = width
                else:
                    right = int(countX * self.chunkXRes)
                bbox = (left, upper, right, lower)
                working_slice = image.crop(bbox)
                slicedImg.append(working_slice)
                left += self.chunkYRes
                countX += 1
            upper += self.chunkXRes
            countY += 1
        return slicedImg

    def createTrainingData(self, sizeOf, category):
        """
        Appends chunked color masked images to a list
        """
        dataDir = "C:/"
        path = os.path.join(dataDir, category)
        for img in os.listdir(path)[:int(len(os.listdir(path))/sizeOf)]:
            try:
                compCounter = 0
                imgArray = cv2.imread(os.path.join(path, img))
                imgArray = cv2.cvtColor(imgArray, cv2.COLOR_RGB2BGR)
                resizeImg = cv2.resize(imgArray, (self.fullXRes, self.fullYRes))
                #Add a R, G, and B masked version fo the image to the training set for better learning
                while compCounter < 4:
                    maskColor = None
                    if compCounter == 1:
                        maskColor = (255, 0, 0)
                    if compCounter == 2:
                        maskColor = (0, 255, 0)
                    if compCounter == 3:
                        maskColor = (0, 0, 255)
                    chunkedData = []
                    coloredImage = self.colorImageComposite(Image.fromarray(resizeImg), maskColor)
                    chunkOfImgs = self.imgSlice(coloredImage)
                    for i in chunkOfImgs:
                        chunkedData.append(np.asarray(i))
                    self.trainingData.append(chunkedData)
                    compCounter += 1
            except cv2.error as _:
                pass

    def setupData(self, category, sizeOf=1):
        """
        Reading the images into an array that is usable
        """
        print("Processing Pictures.")
        self.createTrainingData(sizeOf, category)
        self.trainingData = self.smartShuffle(4)
        images = []
        #Setting up list with chunks many list inside
        for chunk in range(self.chunks):
            images.append([])
        #Separating the images in chunks for each repespective inner list
        for chunkedArr in self.trainingData:
            chunkNum = 0
            for chunkedImg in chunkedArr:
                images[chunkNum].append(chunkedImg)
                chunkNum += 1
        #Reshaping the images
        for chunk in range(self.chunks):
            images[chunk] = np.array(images[chunk]).reshape(-1, self.chunkXRes, self.chunkYRes, 3)
        print("Pictures Processed.")
        return images

    def stitchChunkImg(self, chunkImgList):
        """
        Takes the chunked images and stitches them into a full image
        """
        result = Image.new('RGB', (self.fullXRes, self.fullYRes), 255)
        row = 0
        col = 0
        nextRow = int(math.sqrt(self.chunks))
        for img in chunkImgList:
            result.paste(im=img, box=(col*self.chunkXRes, row*self.chunkYRes))
            col += 1
            if col == nextRow:
                col = 0
                row += 1
        return result

    def colorImageComposite(self, image, color):
        """
        Takes in PIL image object to be mixed with the input color
        which is a 1x3 tuple with 0-1 floating point RGB values
        """
        alpha = 0.3
        if color is None:
            alpha = 0
            colorImg = Image.new("RGB", (self.fullXRes, self.fullYRes), (0, 0, 0))
        else:
            colorImg = Image.new("RGB", (self.fullXRes, self.fullYRes), color)
        compImage = Image.blend(image, colorImg, alpha)
        return compImage

    def smartShuffle(self, numColorMasks):
        """
        Takes in the img data set and shuffles the imgData
        then the order of RGB for each imageData set returning a new list
        """
        print("Shuffling")
        retList = []
        imgDataAsOne = []
        colorDataAsOne = []
        innerLst = 0

        #Creating the list structure
        for _ in range(numColorMasks):
            imgDataAsOne.append([])
        for _ in range(numColorMasks):
            colorDataAsOne.append([])

        #Create a list to keep the images with the same color-alpha filter applied together
        for img in self.trainingData:
            imgDataAsOne[innerLst].append(img)
            innerLst += 1
            if innerLst == numColorMasks:
                innerLst = 0

        #Shuffle imgDataAsOne
        for color in range(numColorMasks):
            random.shuffle(imgDataAsOne[color])

        #Bring the composited images back into a normal list while keeping orignal order default, r, g, b and having shuffled the image data
        for imgData in range(int(len(self.trainingData)/numColorMasks)):
            for colorLst in imgDataAsOne:
                retList.append(colorLst[imgData])
        innerLst = 0

        #Create a list that keeps the images in inner lists of the same color mask
        for img in retList:
            colorDataAsOne[innerLst].append(img)
            innerLst += 1
            if innerLst == numColorMasks:
                innerLst = 0
        retList.clear()

        #Append the images to the final list
        for imgData in range(int(len(self.trainingData)/numColorMasks)):
            random.shuffle(colorDataAsOne)
            colorDataEnum = enumerate(colorDataAsOne)
            for _ in colorDataEnum:
                retList.append(_[1][imgData])
        return retList

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
        self.discriminatorModel.add(tf.keras.layers.Dense(64, activation="relu"))
        self.discriminatorModel.add(tf.keras.layers.Dropout(0.4))
        self.discriminatorModel.add(tf.keras.layers.Dense(50, activation="relu"))
        self.discriminatorModel.add(tf.keras.layers.Dropout(0.4))
        self.discriminatorModel.add(tf.keras.layers.Dense(10, activation="relu"))
        self.discriminatorModel.add(tf.keras.layers.Dropout(0.4))
        self.discriminatorModel.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    def makeGeneratorModel(self):
        """
        Making the generator model
        """
        self.generatorModel = tf.keras.Sequential()
        #7*7*256 for 28x28 img and 14*14*256 for 56x56 image so 1 for a 4x4 img
        self.generatorModel.add(tf.keras.layers.Dense(int(self.chunkXRes/4)*int(self.chunkYRes/4)*256, input_shape=(self.latentSize,)))
        self.generatorModel.add(tf.keras.layers.BatchNormalization())
        self.generatorModel.add(tf.keras.layers.LeakyReLU())
        self.generatorModel.add(tf.keras.layers.Reshape((int(self.chunkXRes/4), int(self.chunkYRes/4), 256)))
        self.generatorModel.add(tf.keras.layers.Conv2DTranspose(128, (3, 3), padding="same"))
        self.generatorModel.add(tf.keras.layers.BatchNormalization())
        self.generatorModel.add(tf.keras.layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same"))
        self.generatorModel.add(tf.keras.layers.BatchNormalization())
        self.generatorModel.add(tf.keras.layers.Conv2DTranspose(3, (3, 3), strides=(2, 2), padding="same", activation="relu"))

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
                self.trainStep(images, np.loadtxt(fr"C:\Coding\Python\ML\GAN\latentSpace\noise{epoch}.txt"))
                latentPoint += 1
            end = timer()
            print(f"Chunk Time Left: {str(datetime.timedelta(seconds=int((self.EPOCHS-(epoch+1))*(end-start))))}")
            print(f"Chunk {self.trainingChunk+1}, Epoch {epoch+1} at {self.chunkXRes}x{self.chunkYRes} finished in: {(end-start):.4f} sec")
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
        self.discriminatorModel.save(fr"C:\Coding\Python\ML\GAN\ganModels\HR\DISC\chunk{self.trainingChunk+1}of{self.chunks}"+
                                     fr"landscapeDISC{self.fullXRes}x{self.fullYRes}res-{self.EPOCHS}epochs-{self.latentSize}latent.model")
        self.generatorModel.save(fr"C:\Coding\Python\ML\GAN\ganModels\HR\GEN\chunk{self.trainingChunk+1}of{self.chunks}"+
                                 fr"landscapeGEN{self.fullXRes}x{self.fullYRes}res-{self.EPOCHS}epochs-{self.latentSize}latent.model")
        fullStop = timer()
        print(f"Training {self.EPOCHS} epochs finished in: {str(datetime.timedelta(seconds=int(fullStop-fullStart)))}")

        #Show image if training for all chunks is finished
        if self.trainingChunk+1 == self.chunks:
            with open(r"C:\Coding\Python\ML\GAN\\trainingChunk.txt", "w") as trainingChunkFile:
                json.dump(-1, trainingChunkFile)
            again = input("Press Enter to see the results: ")
            while again == "":
                latentPoint = np.random.randn(1, self.latentSize)
                chunkImgList = []
                for chunk in range(self.chunks):
                    generatingModel = tf.keras.models.load_model(fr"C:\Coding\Python\ML\\GAN\ganModels\HR\GEN\chunk{chunk+1}of{self.chunks}"+
                                                                 fr"landscapeGEN{self.fullXRes}x{self.fullYRes}res-{self.EPOCHS}epochs-{self.latentSize}"+
                                                                 r"latent.model", compile=False)
                    imgArr = np.asarray(tf.reshape(generatingModel(latentPoint), (self.chunkXRes, self.chunkYRes, 3)))
                    chunkImgList.append(Image.fromarray((imgArr * 255).astype(np.uint8)))
                    tf.keras.backend.clear_session()
                plt.imshow(Image.fromarray(np.asarray(self.stitchChunkImg(chunkImgList))))
                plt.show()
                again = input("Press Enter to see the results: ")


#No-self-use functions
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


#Variables that the user would want to change for the output.
commonVariables = common.Common()

#Naming the AI
CAS = GAN(commonVariables)

#Running the code
CAS.main()


'''
save and apply weights instead of saving and loading models
fix the image loading into the file so the whole set of images is never all in the RAM like it is near the end of setupData
add the ability to load instead of make new models on the first chunk
then load the overall discriminator and pass the stitched images that were stored in a list to be discriminated like 0
training looks like the colors were applied differetly to the chunks check that
separate into setup data, train models, and use models in different files
'''
