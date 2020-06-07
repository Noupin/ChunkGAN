#pylint: disable=C0103
"""
Ensures the HighResGAN.py runs for all chunks and
finishes traingin of all chunks.
"""
__author__ = "Noupin, KeeKee"


import sys
import subprocess
import datetime
from timeit import default_timer as timer
import json

totalRunTimes = sys.argv[1]
fullStart = timer()

for i in range(0, int(totalRunTimes)):
    chunkStart = timer()
    try:
        subprocess.run(['python', r"C:\Coding\Python\ML\GAN\HR_GAN\HR_GAN.py"], check=True)
    except subprocess.CalledProcessError:
        with open(r"C:\Coding\Python\ML\GAN\HR_GAN\trainingChunk.txt", "w") as trainingChunkFile:
            json.dump(-1, trainingChunkFile)
    chunkStop = timer()

    if int(totalRunTimes)-i > 1:
        print(f"With {int(totalRunTimes)-i} chunks left it will take: "+
              f"{datetime.timedelta(seconds=int((int(totalRunTimes)-i)*(chunkStop-chunkStart)))}")

fullStop = timer()

print(f"All training and viewing took: {datetime.timedelta(seconds=int(fullStop-fullStart))}")

sys.exit()
