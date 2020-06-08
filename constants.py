#pylint: disable=C0103
"""
A model for the commonly used constanst variables in HR_GAN
"""
__author__ = "Noupin"


class Constants:
    """
    Data holding class for configurable settings
    """
    datasetDir = r"C:\Datasets\Images/"
    #Folder paths continuing from datasetDir
    astro = "astroPics"
    cat = "Cat"
    dog = "Dog"
    kk = "KeeKeeKats"
    faces = "faces"
    landscapes = "Landscapes"
    cancer1 = r"Cancer\ACRIN-NSCLC-FDG-PET\ACRIN-NSCLC-FDG-PET-001\01-10-1960-Thorax01 CHEST .75 COLLIMATOR-55959\2.000000-CHEST ROUTINE  5.0  B20f-18218"

    datasetSizePath = r"C:\Coding\Python\ML\GAN\HR_GAN\datasetSize.txt"
    mainPath = r"C:\Coding\Python\ML\GAN\HR_GAN\HR_GAN.py"
    trainingChunkPath = r"C:\Coding\Python\ML\GAN\HR_GAN\trainingChunk.txt"
