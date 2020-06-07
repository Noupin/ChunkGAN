#pylint: disable=C0103, E1101, C0301, W0105
"""
Makes, trains and shows the images from a GAN. The generated
images can be chunked to allowe for better perfromance and
higher resolution generated images than with a single model.
"""
__author__ = "Noupin"


#First Party Imports
import train
import display
from preprocess import Preprocess


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

        #Loading the preprocessed data
        preprocessVars = Preprocess()

        #The training and display of the trained models
        self.modelTrain = train.Train(preprocessVars)
        self.disp = display.Display(preprocessVars)

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
then load the overall discriminator and pass the stitched images that were stored in a list to be discriminated like 0
'''
