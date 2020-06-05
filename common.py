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
        super().__init__()
        self.fullIMGXRes = 128
        self.fullIMGYRes = 128
        self.chunkIMGXRes = 128
        self.chunkIMGYRes = 128
        self.totalEPOCHS = 100
