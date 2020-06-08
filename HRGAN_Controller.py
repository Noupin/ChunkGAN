#pylint: disable=C0103
"""
Ensures the HighResGAN.py runs for all chunks and
finishes traingin of all chunks.
"""
__author__ = "Noupin, KeeKee"

#Third Party Imports
import sys
import subprocess
import datetime
from timeit import default_timer as timer
import json
import wandb

#First Party Imports
from constants import Constants
from tunableVariables import Tunable


totalRunTimes = Tunable.chunks
fullStart = timer()

for i in range(0, int(totalRunTimes)):
    chunkStart = timer()
    try:
        subprocess.run(['python', Constants.mainPath], check=True)
    except subprocess.CalledProcessError:
        with open(r"C:\Coding\Python\ML\GAN\HR_GAN\trainingChunk.txt", "w") as trainingChunkFile:
            json.dump(-1, trainingChunkFile)
            break
    chunkStop = timer()

    if int(totalRunTimes)-i > 1:
        print(f"With {int(totalRunTimes)-(i+1)} chunks left it will take: "+
              f"{datetime.timedelta(seconds=int((int(totalRunTimes)-i)*(chunkStop-chunkStart)))}")

fullStop = timer()

print(f"All training and viewing took: {datetime.timedelta(seconds=int(fullStop-fullStart))}")

sys.exit()
