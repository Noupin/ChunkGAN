__author__ = "Noupin"

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


class Preprocess():
    """
    Prepares the images for use to train the AI model
    """
    
    def __init__(self, commonVars):
        """
        Initialization of all variables needed
        """

        #Resolutions & Amount of Chunks
        self.fullXRes = commonVars.fullIMGXRes
        self.fullYRes = commonVars.fullIMGYRes
        self.chunkXRes = commonVars.chunkIMGXRes
        self.chunkYRes = commonVars.chunkIMGYRes
        self.chunks = commonVars.chunks

        #Getting the current training chunk and setting up the next trainging chunk
        self.trainingChunk = None
        with open(r"C:\Coding\Python\ML\GAN\HR_GAN\trainingChunk.txt") as trainingChunkFile:
            self.trainingChunk = json.load(trainingChunkFile) + 1
        with open(r"C:\Coding\Python\ML\GAN\HR_GAN\trainingChunk.txt", "w") as trainingChunkFile:
            json.dump(self.trainingChunk, trainingChunkFile)

        #Dataset Paths
        astro = "astroPics"
        cat = "Cat"
        dog = "Dog"
        kk = "KeeKeeKats"
        faces = "faces"
        landscapes = "Landscapes"

        #List of Datasets
        self.lstOfDatasets = [astro, cat, dog, faces, kk, landscapes]

        #Setting the number of EPOCHS and size of latent space
        self.EPOCHS = commonVars.totalEPOCHS
        self.latentSize = commonVars.latentSize

        #Setting up chosen dataset to be used
        self.trainingData = []
        self.createDataset()

        #Get the size of the given dataset
        with open(r"C:\Coding\Python\ML\GAN\HR_GAN\datasetSize.txt") as imgAmountFile:
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
        
        self.createLatentPoints()

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

    def addMaskedImgs(self, sizeOf, category):
        """
        Appends chunked color masked images to a list
        """
        dataDir = r"C:\Datasets\Images/"
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
            except cv2.error:
                pass

    def setupData(self, category, sizeOf=1):
        """
        Reading the images into an array that is usable
        """
        print("Processing Pictures.")
        self.addMaskedImgs(sizeOf, category)
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

    def createDataset(self):
        """
        Creating the dataset from a given filepath 
        pointing to a folder of images
        """
        if self.trainingChunk == 0: #Only setup and chunk images on the first chunk
            trainImages = self.setupData(self.lstOfDatasets[0], sizeOf=1)
            print(f"\nStoring {str(int(trainImages[0].shape[0]))} Pictures.")

            #Dump the dataset size so it can be known before the dataset is loaded
            with open(r"C:\Coding\Python\ML\GAN\HR_GAN\datasetSize.txt", "w") as datasetSizeFile:
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

    def createLatentPoints(self):
        """
        Creating and writing the latent arrays to file son the first chunk
        """
        if self.trainingChunk == 0:#Only create latent space on first chunks to keep training consistent
            print(f"\nCreating {int((self.datasetSize/self.BATCH_SIZE)*self.EPOCHS)} Latent Training Points.")
            for i in range(int((self.datasetSize/self.BATCH_SIZE)*self.EPOCHS)):#Total trainSteps = (Dataset size or total loaded images/BATCH_SIZE)*epochs
                np.savetxt(fr"C:\Coding\Python\ML\GAN\HR_GAN\latentSpace\noise{i}.txt", np.random.randn(self.BATCH_SIZE, self.latentSize))#Save random latent point in file
            print("Finished Creating Latenet Training Points.")

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
