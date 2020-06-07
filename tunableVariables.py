#pylint: disable=C0103
"""
A model for the commonly used tunable variables in HR_GAN
"""
__author__ = "Noupin"


class Tunable:
    """
    Data holding class for configurable settings
    """
    fullIMGXRes = 256
    fullIMGYRes = 256

    chunkIMGXRes = 256
    chunkIMGYRes = 256

    totalEPOCHS = 150

    latentSize = 50

    chunks = int((fullIMGXRes**2)/(chunkIMGXRes**2))

    dropout = 0.4
    convFilters = 7
    