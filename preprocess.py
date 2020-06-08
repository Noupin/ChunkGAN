#pylint: disable=C0103, C0301, E1101
"""
Preprocess the images to be used in training
"""
__author__ = "Noupin"

#Third Party Imports
import os
import random
import math
import json
import cv2
from PIL import Image
import tensorflow as tf
import numpy as np
import pydicom

import matplotlib.pyplot as plt

#First Party Imports
from tunableVariables import Tunable
from constants import Constants
import utilities

class Preprocess():
    """
    Prepares the images for use to train the AI model
    """

    def __init__(self):
        """
        Initialization of all variables needed
        """

        self.fullXRes = Tunable.fullIMGXRes
        self.fullYRes = Tunable.fullIMGYRes
        self.chunkXRes = Tunable.chunkIMGXRes
        self.chunkYRes = Tunable.chunkIMGYRes
        self.chunks = Tunable.chunks

        self.trainingChunk = None
        with open(r"C:\Coding\Python\ML\GAN\HR_GAN\trainingChunk.txt") as trainingChunkFile:
            self.trainingChunk = json.load(trainingChunkFile) + 1
        with open(r"C:\Coding\Python\ML\GAN\HR_GAN\trainingChunk.txt", "w") as trainingChunkFile:
            json.dump(self.trainingChunk, trainingChunkFile)

        self.lstOfDatasets = [Constants.astro, Constants.cat, Constants.dog, Constants.faces, Constants.kk, Constants.landscapes, Constants.cancer1]

        self.EPOCHS = Tunable.totalEPOCHS
        self.latentSize = Tunable.latentSize

        self.trainingData = []
        self.createDataset()

        with open(Constants.datasetSizePath) as imgAmountFile:
            self.datasetSize = json.load(imgAmountFile)

        if Tunable.colorChannels == 3:
            self.BATCH_SIZE = utilities.factors(int(self.datasetSize/4), cap=Tunable.maxFactor)[-1]*4 #BATCH_SIZE is mulitpled by 4 to account for the 3 color masks and the original img
        if Tunable.colorChannels == 1:
            self.BATCH_SIZE = utilities.factors(int(self.datasetSize), cap=Tunable.maxFactor)[-1]

        self.trainDataset = []
        self.trainImages = []

        print(f"\nLoading {((self.trainingChunk+1)*self.datasetSize)-(self.trainingChunk*self.datasetSize)} Pictures.")

        for img in range(self.trainingChunk*self.datasetSize, (self.trainingChunk+1)*self.datasetSize):
            self.trainImages.append(np.load(fr"D:\ML\chunkedImgArrays\chunkedImg{img}.npy"))

        self.trainDataset = tf.data.Dataset.from_tensor_slices(self.trainImages).batch(int(self.BATCH_SIZE))
        print("Pictures Loaded.")

        if Tunable.colorChannels == 1:
            print(f"\n\nThe Dataset size is: {self.datasetSize} images\n\nThe Chunk size is {self.chunkXRes}x"+
                  f"{self.chunkYRes} with {self.chunks} chunks\n\nThere are {self.datasetSize*self.chunks} total chunked images\n\nThe BATCH_SIZE is: {self.BATCH_SIZE}"+
                  f"\n\nTotal Resolution: {self.chunkXRes}x{self.chunkYRes}\n\n")
        if Tunable.colorChannels == 3:
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
            lower = height if countY == slicesY else int(countY * self.chunkYRes)
            countX = 1
            left = 0
            for _ in range(slicesX):
                right = width if countX == slicesX else int(countX * self.chunkXRes)
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
        path = os.path.join(Constants.datasetDir, category)
        for img in os.listdir(path)[:int(len(os.listdir(path))/sizeOf)]:
            compCounter = 0
            imgArray = cv2.imread(os.path.join(path, img))
            try:
                imgArray = cv2.cvtColor(imgArray, cv2.COLOR_RGB2BGR)
            except cv2.error:
                continue
            resizeImg = cv2.resize(imgArray, (self.fullXRes, self.fullYRes))
            #Add a R, G, and B masked version fo the image to the training set for better learning
            maskColorMap = {0: None, 1: (255, 0, 0), 2: (0, 255, 0), 3: (0, 0, 255)}
            while compCounter < 4:
                chunkedData = []
                coloredImage = self.colorImageComposite(Image.fromarray(resizeImg), maskColorMap[compCounter])
                chunkOfImgs = self.imgSlice(coloredImage)
                for i in chunkOfImgs:
                    chunkedData.append(np.asarray(i))
                self.trainingData.append(chunkedData)
                compCounter += 1
    
    def addGrayImgs(self, sizeOf, category):
        """
        Appends chunked grayscale images to a list
        """
        path = os.path.join(Constants.datasetDir, category)
        for img in os.listdir(path)[:int(len(os.listdir(path))/sizeOf)]:
            imgArray = cv2.imread(os.path.join(path, img))
            try:
                imgArray = cv2.cvtColor(imgArray, cv2.COLOR_RGB2GRAY)
            except cv2.error:
                continue
            resizeImg = cv2.resize(imgArray, (self.fullXRes, self.fullYRes))
            #Add a R, G, and B masked version fo the image to the training set for better learning
            chunkedData = []
            chunkOfImgs = self.imgSlice(Image.fromarray(resizeImg))
            for i in chunkOfImgs:
                chunkedData.append(np.asarray(i))
            self.trainingData.append(chunkedData)
    
    def addCancerImgs(self, sizeOf, category):
        """
        Appends chunked grayscale images to a list
        """
        path = os.path.join(Constants.datasetDir, category)
        for img in os.listdir(path)[:int(len(os.listdir(path))/sizeOf)]:
            imgArray = pydicom.dcmread(os.path.join(path, img))
            imgArray = imgArray.pixel_array
            resizeImg = cv2.resize(imgArray, (self.fullXRes, self.fullYRes))
            #Add a R, G, and B masked version fo the image to the training set for better learning
            chunkedData = []
            chunkOfImgs = self.imgSlice(Image.fromarray(resizeImg))
            for i in chunkOfImgs:
                chunkedData.append(np.asarray(i))
            self.trainingData.append(chunkedData)

    def setupData(self, category, sizeOf=1):
        """
        Reading the images into an array that is usable
        """
        print("Processing Pictures.")
        if Tunable.colorChannels+Tunable.cancerImg == 3:
            self.addMaskedImgs(sizeOf, category)
            self.trainingData = self.smartShuffle(4)
        elif Tunable.colorChannels+Tunable.cancerImg == 2:
            self.addCancerImgs(sizeOf, category)
            self.trainingData = self.smartShuffle(1)
        elif Tunable.colorChannels+Tunable.cancerImg == 1:
            self.addGrayImgs(sizeOf, category)
            self.trainingData = self.smartShuffle(1)
        
        images = []

        for chunk in range(self.chunks):
            images.append([])

        for chunkedArr in self.trainingData:
            chunkNum = 0
            for chunkedImg in chunkedArr:
                images[chunkNum].append(chunkedImg)
                chunkNum += 1

        for chunk in range(self.chunks):
            images[chunk] = np.array(images[chunk]).reshape(-1, self.chunkXRes, self.chunkYRes, Tunable.colorChannels)
        
        print("Pictures Processed.")

        return images

    def createDataset(self):
        """
        Creating the dataset from a given filepath
        pointing to a folder of images
        """
        if not self.trainingChunk == 0:
            return
        
        trainImages = self.setupData(self.lstOfDatasets[Tunable.imgType], sizeOf=Tunable.divideDatasetBy)
        print(f"\nStoring {str(int(trainImages[0].shape[0]))} Pictures.")
        
        with open(Constants.datasetSizePath, "w") as datasetSizeFile:
            json.dump(int(trainImages[0].shape[0]), datasetSizeFile)

        #Normalizing the data to be used
        for chunk in range(self.chunks):
            #Bring colors from 255 base to 0-1 floating point base
            trainImages[chunk] = trainImages[chunk]/255.0
            trainImages[chunk] = trainImages[chunk].reshape(trainImages[0].shape[0], self.chunkXRes, self.chunkYRes, Tunable.colorChannels)

        for chunk in range(self.chunks):
            for img in range(int(trainImages[0].shape[0])):
                #Dump each chunk into a folder sequentially
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

        for _ in range(numColorMasks):
            imgDataAsOne.append([])
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

        #Bring the composited images back into a normal list while keeping 
        #original order default, r, g, b and having shuffled the image data
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
            for color in colorDataEnum:
                retList.append(color[1][imgData])
        return retList

    def createLatentPoints(self):
        """
        Creating and writing the latent arrays to file son the first chunk
        """
        if not self.trainingChunk == 0:
            return
        
        print(f"\nCreating {int((self.datasetSize/self.BATCH_SIZE)*self.EPOCHS)} Latent Training Points.")
        for i in range(int((self.datasetSize/self.BATCH_SIZE)*self.EPOCHS)): #Total trainSteps
            np.savetxt(fr"C:\Coding\Python\ML\GAN\HR_GAN\latentSpace\noise{i}.txt", np.random.randn(self.BATCH_SIZE, self.latentSize))
        print("Finished Creating Latenet Training Points.")
