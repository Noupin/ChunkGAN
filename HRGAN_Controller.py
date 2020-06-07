#pylint: disable=C0103
"""
Ensures the HighResGAN.py runs for all chunks and
finishes traingin of all chunks.
"""

import sys
import subprocess
import datetime
from timeit import default_timer as timer

totalRunTimes = sys.argv[1]

fullStart = timer()

for i in range(0, int(totalRunTimes)):
    chunkStart = timer()
    subprocess.run(['python', r"C:\Coding\Python\ML\GAN\HRGAN.py"], check=True)
    chunkStop = timer()

    print(f"With {int(totalRunTimes)-i} chunks left it will take: "+
          f"{datetime.timedelta(seconds=int((int(totalRunTimes)-i)*(chunkStop-chunkStart)))}")

fullStop = timer()

print(f"All training took: {datetime.timedelta(seconds=int(fullStop-fullStart))}")

sys.exit()
