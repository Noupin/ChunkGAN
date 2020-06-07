__author__ = "Noupin"

#pylint: disable=C0103, E1101, C0301, W0105
"""
Makes, trains and shows the images from a GAN. The generated
images can be chunked to allowe for better perfromance and
higher resolution generated images than with a single model.
"""
#Refactor for Preprocessing, Training, DisplayGenIMG


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
import common
import preprocess
import train
import display


class Master():
    """
    Given a full image resolution, chunked image resolution and
    a number of epochs, GAN will make the models needed, train them
    and once the training for all chunks are finished output as
    many final images as needed.
    """

    def __init__(self):
        """
        Initialization of all variables needed
        """

        #480p 2.39:1 720x302
        #2048x2048 is more than 7.3GB of vRAM for the Master DISC model

        #Variables that the user would want to change for the output.
        self.commonVars = common.Common()
        self.preproVars = preprocess.Preprocess(self.commonVars)

        #The training and display of the trained models
        self.modelTrain = train.Train(self.commonVars, self.preproVars)
        self.disp = display.Display(self.commonVars, self.preproVars)
    
    def main(self):
        """
        Uses the first party imports and executes the code to be use together
        """

        #Train the GEN and DISC
        self.modelTrain.main()
        self.disp.show()


#Naming the AI
CAS = Master()

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
