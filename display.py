#pylint: disable=C0103, C0301
"""
Displays the final images after stiching the chunked images together
"""
__author__ = "Noupin"

#Third Party Imports
import math
import json
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#First Party Imports
from constants import Constants
from preprocess import Preprocess
from tunableVariables import Tunable

class Display():
    """
    Combination of display functions to use in other files
    """

    def __init__(self, preprocessVars):
        """
        Initialization of all variables needed
        """

        #Resolutions & Amount of Chunks
        self.fullXRes = Tunable.fullIMGXRes
        self.fullYRes = Tunable.fullIMGYRes
        self.chunkXRes = Tunable.chunkIMGXRes
        self.chunkYRes = Tunable.chunkIMGYRes
        self.chunks = Tunable.chunks

        #Setting the number of EPOCHS and size of latent space
        self.EPOCHS = Tunable.totalEPOCHS
        self.latentSize = Tunable.latentSize

        #Setting the training dataset and the current chunk
        self.trainDataset = preprocessVars.trainDataset
        self.trainingChunk = preprocessVars.trainingChunk

    def stitchChunkImg(self, chunkImgList):
        """
        Takes the chunked images and stitches them into a full image
        """

        colorTypeMap = {1: "L", 3: "RGB"}
        result = Image.new(colorTypeMap[Tunable.colorChannels], (self.fullXRes, self.fullYRes), 255)
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

    def show(self):
        """
        Show image if training for all chunks is finished
        """

        if not self.trainingChunk+1 == self.chunks:
            return
        
        with open(r"C:\Coding\Python\ML\GAN\HR_GAN\trainingChunk.txt", "w") as trainingChunkFile:
            json.dump(-1, trainingChunkFile)
        again = input("Press Enter to see the results: ")
        while again == "":
            latentPoint = np.random.randn(1, self.latentSize)
            chunkImgList = []
            for chunk in range(self.chunks):
                generatingModel = tf.keras.models.load_model(fr"C:\Coding\Models\ganModels\HighRes\GEN\chunk{chunk+1}of{self.chunks}"+
                                                                fr"landscapeGEN{self.fullXRes}x{self.fullYRes}res-{self.EPOCHS}epochs-{self.latentSize}"+
                                                                r"latent.model", compile=False)
                if Tunable.colorChannels == 3:
                    imgArr = np.asarray(tf.reshape(generatingModel(latentPoint), (self.chunkXRes, self.chunkYRes, Tunable.colorChannels)))
                elif Tunable.colorChannels == 1:
                    imgArr = np.asarray(tf.reshape(generatingModel(latentPoint), (self.chunkXRes, self.chunkYRes)))
                cmapMap = {1: "gray", 2: plt.cm.bone, 3: None}
                chunkImgList.append(Image.fromarray((imgArr * 255).astype(np.uint8)))
                tf.keras.backend.clear_session()
            plt.imshow(Image.fromarray(np.asarray(self.stitchChunkImg(chunkImgList))), cmap=cmapMap[Tunable.colorChannels+Tunable.cancerImg])
            plt.show()
            again = input("Press Enter to see the results: ")
