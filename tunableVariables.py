#pylint: disable=C0103
"""
A model for the commonly used tunable variables in HR_GAN
"""
__author__ = "Noupin"


class Tunable:
    """
    Data holding class for configurable settings
    """
    fullIMGXRes = 128
    fullIMGYRes = 128

    chunkIMGXRes = 128
    chunkIMGYRes = 128

    totalEPOCHS = 500

    latentSize = 100

    chunks = int((fullIMGXRes**2)/(chunkIMGXRes**2))

    imgType = 6 #[astro, cat, dog, faces, kk, landscapes, cancer1]
    colorChannels = 1 #1 = Grayscale, 3 = RGB *Cancer images are always trained in grayscale*
    cancerImg = 1 #0 for no 1 for yes

    divideDatasetBy = 1 #Divide amount of images in datset by this constant
    maxFactor = 50 #Maximum number of images per batch

    dropout = 0.4
    convFilters = 24
    