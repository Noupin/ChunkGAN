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

class Display():
    """
    Combination of display functions to use in other files
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

    def show(self):
        """
        Show image if training for all chunks is finished
        """

        if self.trainingChunk+1 == self.chunks:
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
                    imgArr = np.asarray(tf.reshape(generatingModel(latentPoint), (self.chunkXRes, self.chunkYRes, 3)))
                    chunkImgList.append(Image.fromarray((imgArr * 255).astype(np.uint8)))
                    tf.keras.backend.clear_session()
                plt.imshow(Image.fromarray(np.asarray(self.stitchChunkImg(chunkImgList))))
                plt.show()
                again = input("Press Enter to see the results: ")
