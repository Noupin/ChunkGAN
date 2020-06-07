__author__ = "Noupin"

#pylint: disable=C0103
"""
A model for the commonly used variables in HighResGAN.py
"""

class Common():
    """
    Data holding class or model for common variables
    """

    def __init__(self):
        """
        Initializing the variables
        """

        self.fullIMGXRes = 128
        self.fullIMGYRes = 128
        self.chunkIMGXRes = 64
        self.chunkIMGYRes = 64
        self.totalEPOCHS = 50
        self.latentSize = 100
        self.chunks = int((self.fullIMGXRes**2)/(self.chunkIMGXRes**2))
